"""Direct URL fetch + tool-argument parsing shared across agent toolsets.

``fetch_url_tool`` is the urllib + HTML->text fetch used by the reproduction
agent (``reclaim_repro.tools``); ``parse_tool_arguments`` normalizes the
JSON ``arguments`` payload of a model tool call for every dispatcher.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

from reclaim_vllm.config.config import TOOL_MAX_CHARS, TOOL_TIMEOUT


def http_text(
    url: str,
    timeout: float,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    max_chars: int = 100_000,
) -> tuple[int, str, str, str]:
    request_headers = {
        "User-Agent": "reclaim-artifact-verifier/0.1",
        "Accept": "text/html,application/json,text/plain,*/*",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(max(max_chars * 4, 65536))
            charset = response.headers.get_content_charset() or "utf-8"
            return (
                response.status,
                response.geturl(),
                response.headers.get("Content-Type", ""),
                raw.decode(charset, errors="replace"),
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read(max(max_chars * 4, 65536))
        charset = exc.headers.get_content_charset() or "utf-8"
        return (
            exc.code,
            exc.geturl(),
            exc.headers.get("Content-Type", ""),
            raw.decode(charset, errors="replace"),
        )


def is_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_probably_text(content_type: str) -> bool:
    lowered = content_type.lower()
    return any(marker in lowered for marker in ("text", "html", "json", "xml", "javascript"))


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|li|h[1-6]|tr)>", "\n", html)
    return clean_text(html)


def clean_text(value: str) -> str:
    text = re.sub(r"(?is)<[^>]+>", " ", value)
    text = unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).replace("\n ", "\n").strip()


def fetch_url_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url", "")).strip()
    max_chars = min(int(arguments.get("max_chars") or TOOL_MAX_CHARS), TOOL_MAX_CHARS)
    if not is_http_url(url):
        return {"ok": False, "error": f"Only http(s) URLs are supported: {url}"}
    status, final_url, content_type, text = http_text(url, TOOL_TIMEOUT, max_chars=max_chars)
    body = text if is_probably_text(content_type) else ""
    if "html" in content_type.lower():
        body = html_to_text(body)
    return {
        "ok": 200 <= status < 400,
        "url": url,
        "status": status,
        "final_url": final_url,
        "content_type": content_type,
        "text": body[:max_chars],
    }


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in {None, ""}:
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Tool arguments must be a JSON object, got {value!r}")
