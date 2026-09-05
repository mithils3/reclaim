from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_claude import agent
from reclaim_vllm.runtime import audit_sink
from reclaim_vllm.vllm.io import extracted_response


def _usage(**over) -> SimpleNamespace:
    fields = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    fields.update(over)
    return SimpleNamespace(**fields)


def _text(text: str, **usage) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=_usage(**usage),
        stop_reason="end_turn",
    )


def _tool_call(name: str, arguments: dict, call_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="checking the log"),
            SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments),
        ],
        usage=_usage(),
        stop_reason="tool_use",
    )


class StubClient:
    """Scripted stand-in for anthropic.Anthropic (the SDK is not needed to test the loop)."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(stream=self._stream)

    @contextmanager
    def _stream(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        yield SimpleNamespace(get_final_message=lambda: reply)


VERDICT = {
    "paper_id": "2505.1",
    "central_claim": "c",
    "match_bar_kind": "point_estimate",
    "target_metric": "acc",
    "target_scope": "test",
    "reference_value": 1.0,
    "op": "abs_rel_within",
    "tolerance": 0.05,
    "execution_verified": True,
    "execution_evidence": "commands.log:12",
    "measured_value": 1.0,
    "measured_citation": "out.json",
    "cheat_flags": [],
    "value_comparison": "match",
    "methodology_notes": "",
    "score": 9,
    "confidence": 0.8,
    "rationale": "traced",
}


class RunAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        (self.run_dir / "report.json").write_text('{"reproduced": true}', encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)

    def test_runs_tools_then_returns_the_final_submission(self) -> None:
        client = StubClient([
            _tool_call("read_run_file", {"path": "report.json"}),
            _text("done reading"),
            _text(json.dumps(VERDICT)),
        ])
        result = agent.run_audit(
            client, prompt="grade this", run_dir=self.run_dir, tool_rounds=5
        )
        self.assertEqual(json.loads(result.text)["score"], 9)
        self.assertEqual(result.tool_loop["exit_reason"], "natural")
        self.assertEqual(result.tool_loop["telemetry"], {"tool_calls": 1, "tool_errors": 0})
        # The tool actually ran against the bundle and its result was fed back.
        tool_turn = client.requests[-1]["messages"][2]
        self.assertEqual(tool_turn["role"], "user")
        self.assertIn("reproduced", tool_turn["content"][0]["content"])
        self.assertFalse(tool_turn["content"][0]["is_error"])

    def test_every_request_asks_for_caching_and_thinking(self) -> None:
        client = StubClient([_text(json.dumps(VERDICT)), _text(json.dumps(VERDICT))])
        agent.run_audit(client, prompt="p", run_dir=self.run_dir, effort="high")
        for request in client.requests:
            self.assertEqual(request["cache_control"], {"type": "ephemeral"})
            self.assertEqual(request["thinking"], {"type": "adaptive"})
            self.assertEqual(request["output_config"]["effort"], "high")

    def test_final_turn_is_schema_bound_with_tools_off(self) -> None:
        client = StubClient([_text("thinking out loud"), _text(json.dumps(VERDICT))])
        agent.run_audit(client, prompt="p", run_dir=self.run_dir)
        final = client.requests[-1]
        self.assertEqual(final["tool_choice"], {"type": "none"})
        self.assertEqual(final["output_config"]["format"]["type"], "json_schema")
        self.assertIn("score", final["output_config"]["format"]["schema"]["properties"])

    def test_round_limit_is_reported(self) -> None:
        client = StubClient([
            _tool_call("list_run_files", {}),
            _tool_call("list_run_files", {"path": "."}),
            _text(json.dumps(VERDICT)),
        ])
        result = agent.run_audit(
            client, prompt="p", run_dir=self.run_dir, tool_rounds=2
        )
        self.assertTrue(result.tool_loop["hit_tool_round_limit"])
        self.assertEqual(result.tool_loop["exit_reason"], "round_limit")

    def test_failed_tool_is_returned_as_an_error_result(self) -> None:
        client = StubClient([
            _tool_call("read_run_file", {"path": "../escape"}),
            _text("blocked"),
            _text(json.dumps(VERDICT)),
        ])
        result = agent.run_audit(client, prompt="p", run_dir=self.run_dir)
        self.assertEqual(result.tool_loop["telemetry"]["tool_errors"], 1)
        self.assertTrue(client.requests[-1]["messages"][2]["content"][0]["is_error"])

    def test_refusal_stops_before_the_verdict_turn(self) -> None:
        refusal = _text("")
        refusal.stop_reason = "refusal"
        client = StubClient([refusal])
        result = agent.run_audit(client, prompt="p", run_dir=self.run_dir)
        self.assertEqual(result.tool_loop["exit_reason"], "refusal")
        self.assertEqual(result.text, "")
        self.assertEqual(len(client.requests), 1)

    def test_events_stream_rounds_calls_and_results(self) -> None:
        client = StubClient([
            _tool_call("read_run_file", {"path": "report.json"}),
            _text("done"),
            _text(json.dumps(VERDICT)),
        ])
        seen: list[str] = []
        agent.run_audit(
            client, prompt="p", run_dir=self.run_dir,
            on_event=lambda kind, idx, payload: seen.append(kind),
        )
        self.assertEqual(
            seen,
            ["round_open", "call_start", "call_result", "round_open", "verdict_turn"],
        )

    def test_verdict_row_matches_the_vllm_auditor_finalizer(self) -> None:
        client = StubClient([_text(json.dumps({**VERDICT, "score": 10}))] * 2)
        result = agent.run_audit(client, prompt="p", run_dir=self.run_dir)
        row = extracted_response(
            "2505.1",
            {"response": {"body": {"choices": [{"message": {"content": result.text}}]}},
             "tool_loop": result.tool_loop},
        )
        self.assertEqual(row["verdict"], "reproduced")
        self.assertTrue(row["reproduced"])
        self.assertEqual(row["verification_status"], "verified")


class AuditSinkContractTests(unittest.TestCase):
    """The events the loop emits must render as rows the Audits page can read."""

    def test_round_message_becomes_an_assistant_row(self) -> None:
        message = _tool_call("bash", {"command": "grep -n acc run.log"})
        message.content.insert(0, SimpleNamespace(type="text", text="checking the score"))
        row = audit_sink.message_row(
            {"audit_run_id": "a", "seq": 0, "kind": "round_open", "round_index": 0},
            {"message": agent._round_message(message)},
        )
        self.assertEqual(row["role"], "assistant")
        self.assertEqual(row["content"], "checking the score")
        self.assertEqual(row["reasoning"], "checking the log")

    def test_tool_call_becomes_a_call_row_with_its_command(self) -> None:
        call = SimpleNamespace(id="toolu_9", name="bash", input={"command": "ls -la"})
        row = audit_sink.call_row(
            {"audit_run_id": "a", "seq": 1, "kind": "call_start", "round_index": 0},
            agent._openai_call(call),
        )
        self.assertEqual(row["tool_name"], "bash")
        self.assertEqual((row["detail_kind"], row["command"]), ("command", "ls -la"))

    def test_tool_result_becomes_a_result_row(self) -> None:
        row = audit_sink.result_row(
            {"audit_run_id": "a", "seq": 2, "kind": "call_result", "round_index": 0},
            {"ok": True, "returncode": 0, "stdout": "acc 0.91", "stderr": ""},
        )
        self.assertTrue(row["ok"])
        self.assertEqual(row["stdout"], "acc 0.91")

    def test_final_row_carries_the_exit_reason(self) -> None:
        row = audit_sink.message_row(
            {"audit_run_id": "a", "seq": 3, "kind": "final", "round_index": 2},
            {"message": {"content": "{...}"}, "exit_reason": "natural", "verdict": {"score": 7}},
        )
        self.assertEqual(row["exit_reason"], "natural")


class UsageTests(unittest.TestCase):
    def test_costs_cache_reads_at_a_tenth_of_input(self) -> None:
        usage = agent.Usage()
        usage.add(_usage(input_tokens=1_000_000, output_tokens=0))
        self.assertAlmostEqual(usage.cost, 5.0, places=6)
        cached = agent.Usage()
        cached.add(_usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000))
        self.assertAlmostEqual(cached.cost, 0.5, places=6)

    def test_merge_sums_every_counter(self) -> None:
        a, b = agent.Usage(), agent.Usage()
        a.add(_usage(input_tokens=1, output_tokens=2, cache_read_input_tokens=3))
        b.add(_usage(input_tokens=4, output_tokens=5, cache_creation_input_tokens=6))
        a.merge(b)
        self.assertEqual((a.input, a.output, a.cache_read, a.cache_write), (5, 7, 3, 6))


class PricingTests(unittest.TestCase):
    def test_each_tier_is_priced_from_its_own_base(self) -> None:
        opus = agent.Usage(model="claude-opus-4-8")
        sonnet = agent.Usage(model="claude-sonnet-5")
        for usage in (opus, sonnet):
            usage.add(_usage(input_tokens=1_000_000, output_tokens=0))
        self.assertAlmostEqual(opus.cost, 5.0, places=6)
        self.assertAlmostEqual(sonnet.cost, 3.0, places=6)  # post-2026-08-31 rate

    def test_sonnet_5_intro_pricing_expires(self) -> None:
        before = agent.prices_for("claude-sonnet-5", today="2026-08-31")
        after = agent.prices_for("claude-sonnet-5", today="2026-09-01")
        self.assertEqual((before[0], before[1]), (2e-6, 10e-6))
        self.assertEqual((after[0], after[1]), (3e-6, 15e-6))

    def test_unknown_model_falls_back_to_the_opus_tier(self) -> None:
        self.assertEqual(agent.prices_for("some-new-model"), agent.PRICES["claude-opus-4-8"])

    def test_cache_reads_and_writes_track_the_base_rate(self) -> None:
        _in, _out, write, read = agent.prices_for("claude-haiku-4-5")
        self.assertAlmostEqual(write, 1.25e-6)
        self.assertAlmostEqual(read, 0.1e-6)


class SchemaAndToolTests(unittest.TestCase):
    def test_tools_are_translated_from_the_shared_definitions(self) -> None:
        tools = agent.anthropic_tools()
        self.assertEqual(
            sorted(tool["name"] for tool in tools),
            ["bash", "list_run_files", "read_run_file", "write_run_file"],
        )
        for tool in tools:
            self.assertIn("input_schema", tool)
            self.assertTrue(tool["description"])

    def test_schema_drops_bounds_and_rewrites_type_unions(self) -> None:
        schema = agent.verdict_schema()
        score = schema["properties"]["score"]
        self.assertNotIn("minimum", score)
        self.assertNotIn("maximum", score)
        self.assertEqual(
            schema["properties"]["reference_value"]["anyOf"],
            [{"type": "number"}, {"type": "null"}],
        )
        self.assertNotIn("type", schema["properties"]["reference_value"])
        # required/additionalProperties must survive: structured outputs needs them
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("score", schema["required"])

    def test_unknown_tool_is_an_error_result_not_a_crash(self) -> None:
        self.assertFalse(agent.execute_tool("nope", {}, Path("."))["ok"])


if __name__ == "__main__":
    unittest.main()
