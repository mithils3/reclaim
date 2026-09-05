"""Resolve which served endpoint the runner should talk to.

This is the consumer side of the reclaim_serve seam. The agent core is
provider-agnostic: it only POSTs chat-completions to a base URL, so swapping the
model behind it is purely a URL change. A base URL can come from three places, in
priority order:

1. an explicit ``--vllm-server-url`` flag,
2. the ``RECLAIM_SERVER_URL`` environment variable,
3. an endpoint file named by ``RECLAIM_ENDPOINT_FILE`` (the JSON that
   reclaim_serve publishes; we read its ``base_url`` field).

If none is set, the resolver returns ``None``: there is no embedded server, so the
repro harness renders prompts as a dry run and the auditor runner exits with an
error pointing at ``reclaim_serve``.

Which model id to send in each request is resolved the same way: ask the server
what it serves (``GET /v1/models``) and use the single advertised model, unless an
explicit name is given via ``--served-model-name`` / ``RECLAIM_SERVED_MODEL``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# The endpoint file is reclaim_serve's published contract, so its env-var name and
# its reader live there. We are the consumer side and import them rather than
# re-declaring them (reclaim_serve imports nothing from us, by design).
from reclaim_serve.config import ENV_ENDPOINT_FILE
from reclaim_serve.endpoint import read_base_url
from reclaim_vllm.vllm.retry import with_retries

ENV_SERVER_URL = "RECLAIM_SERVER_URL"
ENV_SERVED_MODEL = "RECLAIM_SERVED_MODEL"
ENV_API_KEY = "RECLAIM_API_KEY"
ENV_OPENROUTER_PROVIDER = "RECLAIM_OPENROUTER_PROVIDER"
ENV_CHAT_TEMPLATE_KWARGS = "RECLAIM_CHAT_TEMPLATE_KWARGS"
ENV_REASONING_EFFORT = "RECLAIM_REASONING_EFFORT"
ENV_CONTEXT_LENGTH = "RECLAIM_CONTEXT_LENGTH"
ENV_NO_TRUNCATE_PROMPT = "RECLAIM_NO_TRUNCATE_PROMPT"
MODELS_FETCH_TIMEOUT = 30.0


def resolve_api_key() -> str | None:
    """Bearer token for an authenticated endpoint (e.g. OpenRouter), or None.

    A local self-served vLLM needs no key, so this is empty by default and the
    request goes out unauthenticated exactly as before. We read only the explicit
    ``RECLAIM_API_KEY`` (or ``OPENROUTER_API_KEY``) — never a key meant for a
    different provider — so a stray provider key in the shell can't leak to a URL
    it wasn't issued for.
    """
    value = os.environ.get(ENV_API_KEY) or os.environ.get("OPENROUTER_API_KEY")
    return (value or "").strip() or None


def auth_headers() -> dict[str, str]:
    """``Authorization: Bearer`` header when a key is configured, else ``{}``."""
    key = resolve_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def openrouter_provider_routing() -> dict[str, Any] | None:
    """OpenRouter ``provider`` routing block that pins one or more upstream providers.

    Set ``RECLAIM_OPENROUTER_PROVIDER`` to an OpenRouter provider slug (e.g.
    ``deepseek``) to force every request to that provider with fallbacks off — so a
    cache-read-dominated run is billed at that provider's own cache pricing and can't
    be silently rerouted to a pricier host. A comma-separated list sets a preference
    order (first available wins, still no fallback beyond the list). Unset/empty ->
    ``None`` and no ``provider`` field is sent: OpenRouter keeps its default routing,
    and a local vLLM (which ignores the field) is unaffected either way.
    """
    raw = (os.environ.get(ENV_OPENROUTER_PROVIDER) or "").strip()
    providers = [slug.strip() for slug in raw.split(",") if slug.strip()]
    if not providers:
        return None
    return {"order": providers, "allow_fallbacks": False}


def chat_template_kwargs() -> dict[str, Any] | None:
    """Per-request ``chat_template_kwargs`` to attach to every completion, or None.

    Set ``RECLAIM_CHAT_TEMPLATE_KWARGS`` to a JSON object and it rides on every
    chat-completion body (repro loop, auditor/classifier tool loop). This is how a
    model's chat template is told to render a non-default mode that no sampling
    field can express — e.g. DeepSeek-V4-Flash Think Max, which the AA index rung
    depends on:

        RECLAIM_CHAT_TEMPLATE_KWARGS='{"thinking": true, "reasoning_effort": "max"}'

    Unset/empty or unparseable -> ``None`` and no field is sent, so every other
    model (and a plain vLLM that ignores the field) is unaffected.
    """
    raw = (os.environ.get(ENV_CHAT_TEMPLATE_KWARGS) or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(
            f"[chat_template_kwargs] ignoring unparseable {ENV_CHAT_TEMPLATE_KWARGS} "
            f"(not JSON): {raw!r}",
            file=sys.stderr,
            flush=True,
        )
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    return parsed


def reasoning_effort() -> str | None:
    """Per-request ``reasoning_effort`` for every completion, or None.

    The OpenAI-compatible spelling of the knob ``chat_template_kwargs`` covers for a
    locally-served model: a hosted reasoning API takes the depth as a top-level body
    field instead of a chat-template flag. Set ``RECLAIM_REASONING_EFFORT`` to the
    value the provider documents (Muse Spark accepts ``xhigh`` for maximum depth;
    OpenAI-style backends take low/medium/high) and it rides on every chat-completion
    body.

    Unset/empty -> ``None`` and no field is sent, so a local vLLM sweep and every
    non-reasoning model are unaffected.
    """
    value = (os.environ.get(ENV_REASONING_EFFORT) or "").strip()
    return value or None


def truncate_prompt_disabled() -> bool:
    """Whether to strip ``truncate_prompt_tokens`` from every chat-completion body.

    ``truncate_prompt_tokens`` is a vLLM extension: the server clips an over-long
    prompt to the input ceiling instead of erroring. A self-hosted vLLM owns the
    field and OpenRouter tolerates it, so it has ridden on every body since the
    harness was written. A strictly-validating hosted API rejects the unknown
    parameter with a 400 on the first call of both loops (the Meta Model API answers
    ``unknown parameter `truncate_prompt_tokens```), and the preflight never catches
    it because that probe only does a GET on /v1/models.

    Set ``RECLAIM_NO_TRUNCATE_PROMPT=1`` for such a backend. The prompt ceiling then
    rests entirely on the harness's own compaction, so pair it with a context length
    the provider actually documents.

    Unset/empty/0/false/no -> ``False`` and the field is sent, so every vLLM and
    OpenRouter sweep is unaffected.
    """
    value = (os.environ.get(ENV_NO_TRUNCATE_PROMPT) or "").strip().lower()
    return value not in ("", "0", "false", "no")


OPENROUTER_HOST = "openrouter.ai"


def is_openrouter(base_url: str) -> bool:
    """Whether ``base_url`` is OpenRouter, and so understands a ``provider`` block.

    ``provider`` is OpenRouter's own body field for upstream routing. Every other
    backend either ignores an unknown key (a self-hosted vLLM) or rejects the whole
    request: the Meta Model API answers 400 ``unknown parameter `provider```, which
    killed the tools-off final pass of both loops while every tool round before it
    succeeded.

    Presence of an API key used to stand in for "this is OpenRouter". That proxy
    holds only while OpenRouter is the sole keyed backend, and it silently stopped
    being true the moment a second hosted API arrived. Ask the URL instead.
    """
    host = (urllib.parse.urlparse(base_url).hostname or "").lower()
    return host == OPENROUTER_HOST or host.endswith(f".{OPENROUTER_HOST}")


def normalize_server_url(value: str) -> str:
    """Strip a trailing slash and a trailing ``/v1`` so callers can append paths."""
    url = value.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def resolve_server_url(cli_value: str | None) -> str | None:
    """Return the normalized base URL to attach to, or None if none is configured."""
    if cli_value:
        return normalize_server_url(cli_value)
    env_url = os.environ.get(ENV_SERVER_URL)
    if env_url:
        return normalize_server_url(env_url)
    endpoint_file = os.environ.get(ENV_ENDPOINT_FILE)
    if endpoint_file:
        url = read_base_url(Path(endpoint_file))
        if url:
            return normalize_server_url(url)
    return None


def fetch_model_cards(base_url: str) -> list[dict]:
    """Return the raw ``/v1/models`` entries the server advertises (may be empty)."""
    url = f"{base_url.rstrip('/')}/v1/models"

    def _do() -> dict:
        request = urllib.request.Request(url, headers=auth_headers(), method="GET")
        with urllib.request.urlopen(request, timeout=MODELS_FETCH_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        data = with_retries(_do, what=f"GET {url}")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not list models at {url}: {exc}") from exc
    entries = data.get("data") if isinstance(data, dict) else None
    return [entry for entry in entries or [] if isinstance(entry, dict)]


def fetch_served_models(base_url: str) -> list[str]:
    """Return the model ids the server advertises at ``/v1/models`` (may be empty)."""
    return [
        entry["id"]
        for entry in fetch_model_cards(base_url)
        if isinstance(entry.get("id"), str) and entry["id"]
    ]


def fetch_served_context_length(base_url: str, model_id: str | None = None) -> int:
    """Context window the server advertises for ``model_id``.

    vLLM's model card carries ``max_model_len``; OpenAI-compatible proxies (OpenRouter)
    carry ``context_length``, so we read either. Raises rather than guessing a default:
    the served window is the harness's input ceiling, and inventing one is how a
    1M-context model ends up capped at some number the server never agreed to.

    ``RECLAIM_CONTEXT_LENGTH`` overrides the lookup for a server whose model cards
    carry no window at all (the Meta Model API advertises only id/object/created/
    owned_by, so Muse Spark cannot be resolved from its card). That is the operator
    stating the ceiling, which is different from this function inventing one, so an
    unset env var still raises rather than falling back to a number nobody chose.
    """
    override = (os.environ.get(ENV_CONTEXT_LENGTH) or "").strip()
    if override:
        try:
            value = int(override)
        except ValueError:
            raise RuntimeError(
                f"{ENV_CONTEXT_LENGTH}={override!r} is not an integer."
            ) from None
        if value <= 0:
            raise RuntimeError(f"{ENV_CONTEXT_LENGTH}={value} must be positive.")
        return value
    for entry in fetch_model_cards(base_url):
        if model_id and entry.get("id") != model_id:
            continue
        for field in ("max_model_len", "context_length"):
            value = entry.get(field)
            if isinstance(value, int) and value > 0:
                return value
    raise RuntimeError(
        f"{base_url}/v1/models advertised no context window for {model_id!r}; "
        f"cannot resolve the input ceiling. Set {ENV_CONTEXT_LENGTH} to the window "
        f"the provider documents if its model cards omit it."
    )


def resolve_served_model(base_url: str, cli_value: str | None = None) -> str:
    """Pick the model id to send in requests against an attached server.

    Priority for the name: ``--served-model-name`` flag > ``RECLAIM_SERVED_MODEL``
    env > the model the server advertises at ``/v1/models``. With no override we use
    the single advertised model (the common case: one model per serve job); if the
    server lists several we take the first and say so. An explicit override is used
    verbatim but checked against the advertised list so a typo fails loudly here
    rather than as a per-request 404.
    """
    available = fetch_served_models(base_url)
    override = (cli_value or os.environ.get(ENV_SERVED_MODEL) or "").strip()
    if override:
        if available and override not in available:
            raise RuntimeError(
                f"requested model {override!r} is not served by {base_url}; "
                f"available: {available}"
            )
        return override
    if not available:
        raise RuntimeError(
            f"{base_url}/v1/models advertised no models; cannot auto-select. "
            f"Pass --served-model-name to choose explicitly."
        )
    if len(available) > 1:
        print(
            f"Server advertises {len(available)} models {available}; using the "
            f"first ({available[0]!r}). Pass --served-model-name to override.",
            file=sys.stderr,
        )
    return available[0]
