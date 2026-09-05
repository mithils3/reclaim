from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.transcript import (
    TRUNCATED_REASONING_KEEP_CHARS,
    WALL_DEADLINE_ENV,
    length_nudge_message,
    round_status_message,
    sweep_wall_note,
    trim_truncated_reasoning,
)


def test_round_status_message_without_wall_env(monkeypatch):
    monkeypatch.delenv(WALL_DEADLINE_ENV, raising=False)
    msg = round_status_message(0, 300)
    assert msg == {"role": "user", "content": "Tool round 1/300 · 299 left"}


def test_sweep_wall_note_hours_and_minutes_left(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setenv(WALL_DEADLINE_ENV, str(now + 47 * 3600 + 59 * 60 + 30))
    assert sweep_wall_note(now=now) == "sweep wall ~47h59m left"


def test_sweep_wall_note_small_remainder(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setenv(WALL_DEADLINE_ENV, str(now + 90))
    assert sweep_wall_note(now=now) == "sweep wall ~0h01m left"


def test_sweep_wall_note_expired(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setenv(WALL_DEADLINE_ENV, str(now - 1))
    note = sweep_wall_note(now=now)
    assert note == "sweep wall EXPIRED — the job hosting this run is being killed; finalize immediately"


def test_sweep_wall_note_garbage_value_returns_none(monkeypatch):
    monkeypatch.setenv(WALL_DEADLINE_ENV, "abc")
    assert sweep_wall_note() is None


def test_sweep_wall_note_unset_returns_none(monkeypatch):
    monkeypatch.delenv(WALL_DEADLINE_ENV, raising=False)
    assert sweep_wall_note() is None


def test_round_status_message_with_wall_env(monkeypatch):
    monkeypatch.setenv(WALL_DEADLINE_ENV, str(time.time() + 3600))
    msg = round_status_message(0, 300)
    assert "sweep wall ~" in msg["content"]
    assert msg["content"].startswith("Tool round 1/300 · 299 left · ")


def test_trim_truncated_reasoning_cuts_long_fields():
    long = "x" * (TRUNCATED_REASONING_KEEP_CHARS + 500)
    message = {"role": "assistant", "reasoning": long, "reasoning_content": long}
    trimmed = trim_truncated_reasoning(message)
    for key in ("reasoning", "reasoning_content"):
        assert len(trimmed[key]) < len(long)
        assert trimmed[key].startswith("x" * TRUNCATED_REASONING_KEEP_CHARS)
        assert "cut off at the output-token limit" in trimmed[key]
    # The input dict is never mutated.
    assert message["reasoning"] == long
    assert message["reasoning_content"] == long


def test_trim_truncated_reasoning_leaves_short_fields():
    message = {"role": "assistant", "reasoning": "short plan", "content": "hi"}
    trimmed = trim_truncated_reasoning(message)
    assert trimmed["reasoning"] == "short plan"
    assert trimmed["content"] == "hi"
    assert trimmed is not message


def test_length_nudge_message_is_user_role():
    msg = length_nudge_message()
    assert msg["role"] == "user"
    assert "tool call" in msg["content"]
