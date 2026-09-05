/* render_round.js: round cards, tool calls, stdout/stderr collapsibles and the
   tool-call json tree. Split out of render.js; attaches its builders onto
   window.RENDER. Reasoning and assistant prose sit on flat cards for legibility;
   the costliest round wears a spike flag; the FINAL round carries the verdict cap
   with spend against the estimate. When a round's content or reasoning (or a
   stdout dump) is a machine-JSON payload, JsonView turns it into a card or tree;
   its delegated handler also drives the tree carets. */
"use strict";

(function () {
  const R = window.RENDER, esc = R.esc, num = R.num;
  const collapsibleHtml = (summary, inner, open) => `<div class="collapse"><details ${open ? "open" : ""}><summary>${esc(summary)}</summary>${inner}</details></div>`;

  // ---- json tree (tool-call args) ----
  const isUrl = (s) => typeof s === "string" && /^https?:\/\/\S+$/.test(s);
  const keyHtml = (k) => k === null ? "" : `<span class="jkey">"${esc(k)}"</span><span class="jpunc">: </span>`;
  const TOG = `<span class="jtoggle"></span>`;
  function valHtml(v) {
    if (v === null) return `<span class="jval jnull">null</span>`;
    const t = typeof v;
    if (t === "boolean") return `<span class="jval jbool">${v}</span>`;
    if (t === "number") return `<span class="jval jnum">${esc(String(v))}</span>`;
    if (isUrl(v)) return `<a class="jval jstr jlink" href="${esc(v)}" target="_blank" rel="noopener">"${esc(v)}"</a>`;
    return `<span class="jval jstr">"${esc(String(v))}"</span>`;
  }
  function nodeHtml(value, key, isLast) {
    const comma = isLast ? "" : `<span class="jpunc">,</span>`;
    if (value === null || typeof value !== "object")
      return `<div class="jnode"><div class="jline">${TOG}${keyHtml(key)}${valHtml(value)}${comma}</div></div>`;
    const arr = Array.isArray(value), open = arr ? "[" : "{", close = arr ? "]" : "}";
    const keys = arr ? value.map((_, i) => i) : Object.keys(value);
    if (!keys.length) return `<div class="jnode"><div class="jline">${TOG}${keyHtml(key)}<span class="jbrace">${open}${close}</span>${comma}</div></div>`;
    const n = keys.length, count = arr ? `${n} item${n > 1 ? "s" : ""}` : `${n} key${n > 1 ? "s" : ""}`;
    const children = keys.map((k, i) => nodeHtml(value[k], arr ? null : k, i === n - 1)).join("");
    return `<div class="jnode branch"><div class="jline jhead">${TOG}${keyHtml(key)}<span class="jbrace">${open}</span><span class="jpreview"> … <span class="jbrace">${close}</span>${comma} <span class="jcount">${count}</span></span></div>` +
      `<div class="jchildren">${children}</div><div class="jline jclose">${TOG}<span class="jbrace">${close}</span>${comma}</div></div>`;
  }

  // ---- calls ----
  function streamHtml(label, text, cls) {
    if (!text) return "";
    const n = text.split("\n").length;
    const jv = text.length < 60000 && window.JsonView && window.JsonView.parse(text);
    if (jv) return collapsibleHtml(`${label} · json · ${n} lines`, `<div class="jv-stream">${window.JsonView.treeHtml(jv)}</div>`, false);
    // uni() after esc() is safe: it decodes nothing below U+00A0, so a captured
    // stream keeps every byte it printed and only a doubly-encoded π comes back
    return collapsibleHtml(`${label} · ${n} line${n !== 1 ? "s" : ""}`, `<pre class="stream ${cls}">${R.uni(esc(text))}</pre>`, false);
  }
  function resultHtml(c) {
    if (c.ok === undefined) return "";
    const chip = (v) => `<span class="schip">${esc(v)}</span>`;
    let s = `<div class="result"><span class="badge ${c.ok ? "yes" : "no"}">${c.ok ? "ok" : "failed"}</span>`;
    if (c.rc != null) s += chip("rc " + c.rc);
    if (c.duration_s != null) s += chip(c.duration_s + "s");
    if (c.cost_h100 != null) s += `<span class="schip cost">+${c.cost_h100} h</span>`;
    if (c.remaining_h100 != null) s += chip(c.remaining_h100 + " left");
    if (c.bytes_written != null) s += chip(c.bytes_written + " B");
    if (c.path && c.detail_kind !== "path") s += chip(c.path);
    return s + "</div>";
  }
  function callHtml(c) {
    let body = "";
    if (c.detail_kind === "json" && c.args) body += `<div class="jraw">${nodeHtml(c.args, null, true)}</div>`;
    else if (c.command) body += `<pre class="cmd ${c.detail_kind || ""}">${esc(c.command)}</pre>`;
    body += resultHtml(c);
    if (c.error) body += `<div class="err">${R.uesc(c.error)}</div>`;
    body += streamHtml("stdout", c.stdout, "out");
    body += streamHtml("stderr", c.stderr, "errt");
    if (c.truncated) body += `<div class="trunc">… output truncated</div>`;
    return `<div class="call"><div class="call-h"><span class="tool">${esc(c.tool_name)}</span></div><div class="call-body">${body}</div></div>`;
  }
  // ---- minimal markdown for the agent's own prose ----------------------------
  // The FINAL/assistant block is the text a reviewer reads closest, and models
  // write it in markdown. Escape FIRST, then transform the escaped string, so a
  // transform can never introduce markup the escape was meant to remove.
  // Supported on purpose and nothing else: ``` fences, ATX headings, **bold**,
  // `inline code`. Everything else rides the .rprose pre-wrap as written.
  const mdInline = (s) => s
    .replace(/`([^`\n]+)`/g, `<code class="rcode-i">$1</code>`)
    .replace(/\*\*(?!\s)([^*]+?)\*\*/g, "<strong>$1</strong>");
  function mdProse(chunk) {
    const lines = chunk.split("\n");
    let out = "", buf = [];
    const flush = () => {
      while (buf.length && !buf[0].trim()) buf.shift();
      while (buf.length && !buf[buf.length - 1].trim()) buf.pop();
      if (buf.length) out += `<p class="rprose">${mdInline(buf.join("\n"))}</p>`;
      buf = [];
    };
    for (const ln of lines) {
      const h = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(ln);
      if (h) { flush(); out += `<h4 class="rhead">${mdInline(h[2])}</h4>`; }
      else buf.push(ln);
    }
    flush();
    return out;
  }
  // a fenced block stays monospaced and unwrapped in its own scroller; splitting
  // on ``` makes every odd chunk the inside of a fence (an unterminated last
  // fence still reads as code, which is what it is)
  function mdHtml(text) {
    const parts = esc(String(text)).split("```");
    let out = "";
    parts.forEach((chunk, i) => {
      // prose is decoded, a fenced block is not: source the agent pasted keeps
      // its literal escapes, the sentence around it reads as written
      if (i % 2 === 0) { out += mdProse(R.uni(chunk)); return; }
      const nl = chunk.indexOf("\n");
      const lang = nl < 0 ? "" : chunk.slice(0, nl).trim();
      const code = (nl < 0 ? chunk : chunk.slice(nl + 1)).replace(/\s+$/, "");
      const tag = /^[a-z0-9+#-]{1,12}$/i.test(lang) ? `<span class="rcode-l">${lang}</span>` : "";
      out += `<div class="rcode-w">${tag}<pre class="rcode">${code}</pre></div>`;
    });
    return out || `<p class="rprose">${R.uesc(text)}</p>`;
  }

  function block(label, text) {
    if (!text) return "";
    const jv = window.JsonView && window.JsonView.blockHtml(text);
    if (jv) return `<div class="block flat jv-block"><div class="block-l">${jv.label || label}</div>${jv.html}</div>`;
    return `<div class="block flat"><div class="block-l">${label}</div><div class="rmd">${mdHtml(text)}</div></div>`;
  }

  function roundCost(round) {
    let cost = 0, rem = null, any = false;
    for (const c of (round.calls || [])) { if (typeof c.cost_h100 === "number") { cost += c.cost_h100; any = true; } if (c.remaining_h100 != null) rem = c.remaining_h100; }
    return { cost: any ? Math.round(cost * 1e6) / 1e6 : null, rem };
  }
  // tool-name tally ("bash ×3 · write_file ×1") for the always-visible summary line
  function toolTally(calls) {
    const m = new Map();
    for (const c of (calls || [])) { const t = c.tool_name || "?"; m.set(t, (m.get(t) || 0) + 1); }
    return m.size ? [...m].map(([t, n]) => `${esc(t)} ×${n}`).join(" · ") : "";
  }
  const faultCount = (calls) => (calls || []).reduce((a, c) => a + (c.ok === false || c.error ? 1 : 0), 0);

  // what the round actually did, in one line. A 70-round transcript collapses to
  // 70 rows whose left end is "ROUND 48 · run_gpu ×1" and whose right end is a
  // cost chip, with the whole middle of the card blank; the gist fills that span
  // with the first command the round ran, so consecutive rows read as different
  // work rather than as filler.
  function oneLine(s, max) {
    const t = String(s == null ? "" : s).replace(/\s+/g, " ").trim();
    return t.length > max ? t.slice(0, max - 1) + "…" : t;
  }
  // the prose fallback is markdown, so its markers come off first; a command is
  // never put through this, because "2>&1" and "*.py" are not markdown
  const oneLineMd = (s, max) => oneLine(String(s == null ? "" : s)
    .replace(/```[a-z0-9+#-]*/gi, " ").replace(/^\s{0,3}#{1,6}\s+/gm, "").replace(/[`*]/g, ""), max);
  function roundGist(round) {
    const c = (round.calls || [])[0];
    if (c) {
      if (c.command) return oneLine(c.command, 150);
      if (c.args && typeof c.args === "object" && !Array.isArray(c.args)) {
        for (const k of Object.keys(c.args)) {
          const v = c.args[k];
          if (typeof v === "string" || typeof v === "number") return oneLine(`${k}: ${v}`, 150);
        }
      }
      if (c.path) return oneLine(c.path, 150);
    }
    // the prose fallback is decoded before it is cut, so a summary line never
    // shows half a backslash-u escape where a character belongs
    return oneLineMd(R.uni(round.content || round.reasoning || ""), 150);
  }

  // a round card is collapsible: an always-visible summary <button> header +
  // a hidden-when-collapsed .rcard-body. FINAL rounds always render expanded.
  function roundCardHtml(round, arxiv, run, collapsed) {
    const V = window.Verdict, isFinal = round.kind === "final";
    const bad = isFinal && round.exit_reason && !R.GOOD_EXIT(round.exit_reason);
    const key = `${round.kind}:${round.round_index}`;
    const { cost, rem } = roundCost(round);
    const spike = run && run.__spike > 0 && cost != null && cost >= run.__spike && cost > 0;
    const exit = isFinal && round.exit_reason ? `<span class="badge ${bad ? "no" : "yes"}">${esc(R.exitLabel(round.exit_reason))}</span>` : "";
    const tally = toolTally(round.calls), faults = faultCount(round.calls);
    const faultFlag = faults ? `<span class="r-fault">✕ ${faults} failed</span>` : "";
    // A 'length' finish means the turn hit the output-token limit and was cut off
    // mid stream. Flag it like the other round markers.
    const cutFlag = round.finish_reason === "length"
      ? `<span class="badge no" title="turn hit the output-token limit">✂ cut off at token limit</span>` : "";
    const costChip = cost != null ? `<span class="r-cost tnum" title="+${cost} H100·h${rem != null ? ` · ${rem} left` : ""}">+${R.fmtHM(cost)}${rem != null ? ` · ${R.fmtHM(rem)} left` : ""}</span>` : "";
    const isCol = !isFinal && !!collapsed;
    const gist = isFinal ? "" : roundGist(round);
    const gistEl = gist ? `<span class="r-what" title="${esc(gist)}">${esc(gist)}</span>` : "";
    let cap = "";
    if (isFinal && V && run) {
      const fam = V.ofRun(run), took = R.num(run.spent_h100), pred = R.predictedOf(run);
      cap = `<div class="verdict-cap ${fam}">${V.inline(fam, V.word(run))}<span class="vc-bal" title="took ${took ?? "?"} vs predicted ${pred ?? "?"} H100·h">took <b class="tnum">${R.fmtHM(took)}</b> vs predicted <b class="tnum">${R.fmtHM(pred)}</b> H100·h</span></div>`;
    }
    return `<div class="rcard ${isFinal ? "final" : ""} ${bad ? "bad" : ""} ${spike ? "spike" : ""} ${faults && !isFinal ? "fault" : ""} ${isCol ? "collapsed" : ""}" data-key="${esc(key)}" data-round="${esc(round.round_index ?? "")}">
      <button class="rcard-h" type="button" aria-expanded="${!isCol}"><span class="r-idx">${isFinal ? `<i class="r-fin">✓</i> FINAL` : "ROUND " + (round.round_index ?? "?")}</span>
        <span class="ftime">${round.ts ? esc(R.fmtTime(round.ts)) : ""}</span>${tally ? `<span class="r-tally">${tally}</span>` : ""}${faultFlag}${cutFlag}${exit}${spike ? `<span class="badge over" title="costliest round">⚡ spike</span>` : ""}${gistEl}${costChip}</button>
      <div class="rcard-body">
        ${block("💭 reasoning", round.reasoning)}${block("🗒 assistant", round.content)}
        ${(round.calls || []).map(callHtml).join("")}${cap}</div></div>`;
  }

  Object.assign(window.RENDER, { roundCardHtml, callHtml, nodeHtml });
})();
