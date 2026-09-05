from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

from reclaim_vllm.vllm.endpoint import (
    auth_headers,
    chat_template_kwargs,
    is_openrouter,
    openrouter_provider_routing,
    reasoning_effort,
    truncate_prompt_disabled,
)
from reclaim_vllm.vllm.retry import with_retries

# Universally-supported degrade for a json_schema response_format the routed
# provider can't honor.
JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


def apply_provider_routing(body: dict[str, Any], base_url: str) -> None:
    """Pin the OpenRouter upstream provider in-place when one is configured.

    Single chokepoint for every chat-completion path (repro loop, classifier/auditor
    tool loop, context compaction), so the pin applies uniformly. No-op when unset,
    and never clobbers a ``provider`` block a caller already placed on the body.

    Gated on the endpoint being OpenRouter, because ``provider`` is its field alone
    and a strict backend 400s on the unknown key. A stray env var in a login shell
    can no longer aim it at the wrong host.
    """
    if not is_openrouter(base_url):
        return
    provider = openrouter_provider_routing()
    if provider is not None:
        body.setdefault("provider", provider)


def apply_chat_template_kwargs(body: dict[str, Any]) -> None:
    """Attach env-configured ``chat_template_kwargs`` in-place, if any.

    Same chokepoint and same never-clobber contract as ``apply_provider_routing``:
    a body that already carries the field wins, so a caller can still override
    per request. No-op when the env var is unset (every non-DeepSeek sweep).
    """
    kwargs = chat_template_kwargs()
    if kwargs is not None:
        body.setdefault("chat_template_kwargs", kwargs)


def apply_reasoning_effort(body: dict[str, Any]) -> None:
    """Attach the env-configured ``reasoning_effort`` in-place, if any.

    Same chokepoint and same never-clobber contract as the two above, so one env var
    sets the depth for the repro loop, the auditor tool loop, and compaction alike.
    No-op when unset (every local-vLLM sweep).
    """
    effort = reasoning_effort()
    if effort is not None:
        body.setdefault("reasoning_effort", effort)


def drop_truncate_prompt_tokens(body: dict[str, Any]) -> None:
    """Strip the vLLM-only ``truncate_prompt_tokens`` in-place when configured.

    Inverse direction to the three above and the same chokepoint: ``io.py`` puts the
    field on every body it builds, and a strictly-validating hosted API 400s on it.
    Removing it here covers the repro loop, the auditor tool loop, and compaction with
    one env var, and leaves the builder honest about what it wants.

    No-op unless ``RECLAIM_NO_TRUNCATE_PROMPT`` is set, so a vLLM sweep keeps its
    server-side truncation.
    """
    if truncate_prompt_disabled():
        body.pop("truncate_prompt_tokens", None)


def prepare_structured_output(body: dict[str, Any], base_url: str) -> None:
    """Make a ``json_schema`` response_format actually enforceable on OpenRouter.

    OpenRouter only *enforces* structured outputs on upstreams that advertise the
    ``structured_outputs`` capability; others silently ignore ``response_format``
    or reject it with a 400 ("This response_format type is unavailable now" --
    e.g. DeepSeek/MiniMax first-party). Marking the schema ``strict`` and setting
    ``provider.require_parameters`` tells OpenRouter to route ONLY to a provider
    that can honor the schema (see the OpenRouter structured-outputs + provider-
    routing docs). If none is reachable it 404s, which the request path degrades
    to ``json_object`` rather than failing the run.

    Gated on the endpoint being OpenRouter, since both halves of this (``strict``
    plus ``provider.require_parameters``) exist only to steer OpenRouter's router.
    Everyone else gets the caller's ``response_format`` untouched: a self-hosted
    vLLM enforces ``json_schema`` natively, and a hosted API that rejects unknown
    keys must never be handed a ``provider`` block. An earlier version gated on
    ``resolve_api_key() is not None``, which treated every keyed backend as
    OpenRouter and made the Meta Model API take a 400 on the tools-off final pass
    of every run.
    """
    if not is_openrouter(base_url):
        return
    response_format = body.get("response_format")
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return
    response_format = dict(response_format)
    schema = response_format.get("json_schema")
    if isinstance(schema, dict):
        # Copy so we never mutate the shared module-level schema constant.
        response_format["json_schema"] = {"strict": True, **schema}
    body["response_format"] = response_format
    provider = dict(body.get("provider") or {})
    provider.setdefault("require_parameters", True)
    body["provider"] = provider


def post_chat_completion_row(
    base_url: str,
    row: dict[str, Any],
    timeout: float,
) -> Any:
    apply_provider_routing(row["body"], base_url)
    apply_chat_template_kwargs(row["body"])
    apply_reasoning_effort(row["body"])
    drop_truncate_prompt_tokens(row["body"])
    prepare_structured_output(row["body"], base_url)
    return post_vllm_chat_completion(base_url, row["body"], timeout)


def response_row(custom_id: str, body: Any) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": body,
        },
    }


def post_vllm_chat_completion(base_url: str, body: dict[str, Any], timeout: float) -> Any:
    try:
        return _post_body(base_url, body, timeout)
    except urllib.error.HTTPError as exc:
        downgraded = downgrade_response_format_on_reject(body, exc)
        if downgraded is None:
            raise
        print(
            f"[response_format] no reachable provider enforces json_schema "
            f"(HTTP {exc.code}); retrying once with json_object",
            file=sys.stderr,
            flush=True,
        )
        return _post_body(base_url, downgraded, timeout)


def _post_body(base_url: str, body: dict[str, Any], timeout: float) -> Any:
    data = json.dumps(body).encode("utf-8")

    def _do() -> Any:
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **auth_headers(),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    return with_retries(_do, what=f"chat.completions POST {base_url}")


def downgrade_response_format_on_reject(
    body: dict[str, Any], exc: urllib.error.HTTPError
) -> dict[str, Any] | None:
    """A copy of ``body`` with a json_schema response_format swapped for
    json_object, or ``None`` when the rejection isn't a structured-output one.

    Two ways OpenRouter refuses a ``json_schema`` request when no reachable
    upstream can enforce it:

    * a routed provider rejects it -- 400 *"This response_format type is
      unavailable now"* (DeepSeek/MiniMax first-party), or
    * ``provider.require_parameters`` (added by ``prepare_structured_output``)
      finds no capable provider -- 404 *"No endpoints found that can handle the
      requested parameters"* / *"No allowed providers are available"*.

    Either, unhandled, kills the whole run (e.g. every audit on its tools-off
    final pass). json_object is universally supported and, paired with our
    tolerant JSON parsing + schema finalize, is a safe degrade; we also drop
    ``require_parameters`` so the retry isn't gated on a capability we no longer
    request. A self-hosted vLLM speaks json_schema natively and never hits this.
    (The caller's prompts already contain the word "json", which json_object
    needs.)
    """
    rf = body.get("response_format")
    if not isinstance(rf, dict) or rf.get("type") != "json_schema":
        return None
    detail = (getattr(exc, "reclaim_body", "") or "").lower()
    provider_rejected = "response_format" in detail and any(
        token in detail for token in ("unavailable", "unsupported", "not support")
    )
    no_capable_provider = exc.code == 404 and (
        "requested parameters" in detail or "allowed providers" in detail
    )
    if not (provider_rejected or no_capable_provider):
        return None
    downgraded = dict(body)
    downgraded["response_format"] = dict(JSON_OBJECT_RESPONSE_FORMAT)
    provider = dict(downgraded.get("provider") or {})
    provider.pop("require_parameters", None)
    if provider:
        downgraded["provider"] = provider
    else:
        downgraded.pop("provider", None)
    return downgraded
