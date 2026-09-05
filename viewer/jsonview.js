/* jsonview.js: turns the machine-generated JSON that closes a reproduction run
   or an audit run into a readable instrument card instead of a wall of text.
   parse() cheaply rejects prose and truncated payloads. blockHtml() sniffs the two
   known shapes, the agent's self-report and the auditor's verdict, and renders
   each as a card, falling back to a collapsible JSON tree for anything else. Also
   home to flagsHtml(), the ONE integrity-flag renderer the whole site uses (the
   run page's audit card calls the same function, so the two can never drift into
   printing a raw JSON dump again), and the one document-level click handler that
   toggles the .jnode carets the render_round.js tree emits. Attaches
   window.JsonView; leans on RENDER.esc and RENDER.nodeHtml. */
"use strict";

(function () {
  const R = () => window.RENDER;
  const esc = (s) => R().esc(s);
  // every prose field on a card goes through uesc, not esc: an auditor payload
  // that was JSON-encoded twice carries its non-ASCII characters as literal
  // backslash-u escapes, and this card is where a reviewer reads them
  const uesc = (s) => R().uesc(s);
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const numStr = (x) => (typeof x === "number" ? String(Math.round(x * 1e6) / 1e6) : String(x));

  // ---- parse: object/array or null ------------------------------------------
  function parse(text) {
    if (typeof text !== "string") return null;
    let s = text.trim();
    const fence = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(s);
    if (fence) s = fence[1].trim();
    const a = s[0], z = s[s.length - 1];
    if ((a !== "{" && a !== "[") || (z !== "}" && z !== "]")) return null;
    try { const v = JSON.parse(s); return (v && typeof v === "object") ? v : null; }
    catch (e) { return null; }
  }

  function treeHtml(obj) { return `<div class="jraw">${R().nodeHtml(obj, null, true)}</div>`; }
  function rawFooter(obj) {
    return `<div class="collapse"><details><summary>raw json</summary>` +
      `<pre class="stream jv-rawjson">${esc(JSON.stringify(obj, null, 2))}</pre></details></div>`;
  }
  // collapsible prose section, "" when empty; reuses the .collapse details look
  function sec(label, text, open) {
    if (text == null || String(text).trim() === "") return "";
    return `<details class="jv-sec"${open ? " open" : ""}><summary>${esc(label)}</summary>` +
      `<p class="rprose">${uesc(text)}</p></details>`;
  }
  // leftover keys (not consumed by a card) rendered as a tree, when any remain
  function leftover(o, consumed) {
    const rest = {}; let any = false;
    for (const k of Object.keys(o)) {
      if (consumed.has(k)) continue;
      const v = o[k];
      if (v == null || v === "" || (Array.isArray(v) && !v.length)) continue;
      rest[k] = v; any = true;
    }
    return any ? `<details class="jv-sec"><summary>other fields</summary>${treeHtml(rest)}</details>` : "";
  }

  // ---- op humanizer + pass/fail --------------------------------------------
  // The recorded op is either one of a handful of codes or, on a comparative
  // claim, a whole sentence stating the bar. Codes get a symbol; a sentence is
  // returned untouched and the caller prints it on its own rule line.
  function humanOp(op, tol) {
    switch (op) {
      case "abs_rel_within": return `within ±${numStr(tol * 100)}% relative`;
      case "abs_within": return `within ±${numStr(tol)}`;
      case "order_of_magnitude_within": return "within an order of magnitude";
      case "ge": case "gte": case ">=": case "≥": return "≥ reference";
      case "le": case "lte": case "<=": case "≤": return "≤ reference";
      case "gt": case ">": return "> reference";
      case "lt": case "<": return "< reference";
      case "eq": case "==": case "=": return "equals reference";
      default:
        if (op == null || String(op).trim() === "") return "";
        return String(op) + (tol != null ? ` · tolerance ${numStr(tol)}` : "");
    }
  }
  // { cls, tag, delta }. Pass or fail only when both values are finite numbers
  function compare(op, m, r, tol) {
    if (!isNum(m) || !isNum(r)) return { cls: "", tag: "", delta: "" };
    const abs = `Δ ${numStr(Math.round(Math.abs(m - r) * 1e6) / 1e6)}`;
    let pass, delta = abs;
    if (op === "abs_rel_within") {
      const rel = Math.abs(m - r) / Math.abs(r);
      pass = rel <= tol; delta = `Δ ${Math.round(rel * 10000) / 100}% rel`;
    } else if (op === "abs_within") {
      pass = Math.abs(m - r) <= tol;
    } else if (op === "order_of_magnitude_within") {
      pass = r !== 0 && Math.abs(m / r) >= 0.1 && Math.abs(m / r) <= 10;
    } else if (op === "ge" || op === "gte" || op === ">=" || op === "≥") {
      pass = m >= r;
    } else if (op === "le" || op === "lte" || op === "<=" || op === "≤") {
      pass = m <= r;
    } else if (op === "gt" || op === ">") {
      pass = m > r;
    } else if (op === "lt" || op === "<") {
      pass = m < r;
    } else if (op === "eq" || op === "==" || op === "=") {
      pass = m === r;
    } else return { cls: "", tag: "", delta: abs };
    return { cls: pass ? "pass" : "fail", tag: pass ? "WITHIN TOLERANCE" : "OUTSIDE TOLERANCE", delta };
  }

  // ---- integrity flags: the one renderer -------------------------------------
  // Every recorded flag is {kind, severity, evidence}. Older payloads used
  // flag/name/code and level/detail, so those alias in. Anything left over is
  // printed as labelled key/value lines: a flag NEVER reaches the DOM as a
  // JSON.stringify dump, which is what the raw-JSON pill bug was.
  const FLAG_KEYS = new Set(["kind", "flag", "name", "code", "severity", "level",
    "evidence", "detail", "note"]);
  function sevClass(s) {
    const v = String(s == null ? "" : s).toLowerCase();
    return v === "high" ? "no" : v.startsWith("med") ? "over" : "slate";
  }
  function flagLine(k, v) {
    const txt = (v && typeof v === "object") ? JSON.stringify(v) : String(v);
    return `<div class="an-flag-kv"><span class="an-flag-k">${esc(String(k).replace(/_/g, " "))}</span>` +
      `<span class="an-flag-v">${uesc(txt)}</span></div>`;
  }
  function flagHtml(raw) {
    const f = (raw && typeof raw === "object" && !Array.isArray(raw)) ? raw : { kind: String(raw) };
    const kind = String(f.kind || f.flag || f.name || f.code || "flag").replace(/_/g, " ");
    const sev = f.severity != null ? f.severity : f.level;
    const ev = f.evidence || f.detail || f.note || "";
    const c = sevClass(sev);
    const rest = Object.keys(f)
      .filter((k) => !FLAG_KEYS.has(k) && f[k] != null && f[k] !== "")
      .map((k) => flagLine(k, f[k])).join("");
    return `<div class="an-flag ${c}">` +
      `<div class="an-flag-h"><span class="an-flag-kind">${uesc(kind)}</span>` +
      `${sev ? `<span class="badge ${c}">${esc(sev)}</span>` : ""}</div>` +
      `${ev ? `<div class="an-flag-ev">${uesc(ev)}</div>` : ""}${rest}</div>`;
  }
  function flagsHtml(flags, emptyNote) {
    if (!Array.isArray(flags) || !flags.length)
      return `<p class="an-noflags">✓ ${esc(emptyNote || "no integrity flags")}</p>`;
    return flags.map(flagHtml).join("");
  }

  // ---- audit verdict card ---------------------------------------------------
  const AUDIT_KEYS = new Set(["paper_id", "central_claim", "match_bar_kind", "target_metric",
    "target_scope", "reference_value", "op", "tolerance", "execution_verified", "execution_evidence",
    "measured_value", "measured_citation", "cheat_flags", "value_comparison", "methodology_notes",
    "score", "confidence", "rationale", "verdict"]);

  function scoreBand(s) { const n = isNum(s) ? s : 0; return n >= 8 ? "yes" : n >= 6 ? "over" : "no"; }
  function auditHead(o) {
    const band = scoreBand(o.score), val = o.score != null ? o.score : 0;
    let h = `<span class="jv-score ${band} tnum">${esc(val)}<i>/10</i></span>`;
    if (o.verdict) h += `<span class="schip jv-verdict">${esc(o.verdict)}</span>`;
    if (isNum(o.confidence)) h += `<span class="schip">confidence ${Math.round(o.confidence * 100)}%</span>`;
    if (o.execution_verified === true) h += `<span class="badge yes">execution verified</span>`;
    else if (o.execution_verified === false) h += `<span class="badge no">execution NOT verified</span>`;
    if (o.match_bar_kind) h += `<span class="schip">${esc(o.match_bar_kind)}</span>`;
    if (o.paper_id) h += `<span class="schip">${esc(o.paper_id)}</span>`;
    return `<div class="jv-head">${h}</div>`;
  }
  // measured against reference. Both fields are optional and either can be a
  // sentence rather than a number, so the strip renders only the cells that
  // carry a value (an absent reference collapses the grid instead of leaving a
  // "·" placeholder box) and a long prose bar drops to its own rule line.
  const has = (v) => v != null && String(v).trim() !== "";
  function cell(v, lab) {
    if (!has(v)) return "";
    const n = isNum(v);
    const s = n ? numStr(v) : String(v);
    return `<div class="jv-cell"><div class="jv-cv ${n ? "num tnum" : "txt"}" title="${uesc(s)}">${uesc(s)}</div>` +
      `<div class="jv-cl">${lab}</div></div>`;
  }
  function compareStrip(o) {
    const hasM = has(o.measured_value), hasR = has(o.reference_value);
    const c = hasR ? compare(o.op, o.measured_value, o.reference_value, o.tolerance) : { cls: "", tag: "", delta: "" };
    const opTxt = humanOp(o.op, o.tolerance);
    const inline = hasM && hasR && opTxt.length > 0 && opTxt.length <= 34;
    const metric = o.target_metric ? `<div class="jv-metric">${uesc(o.target_metric)}</div>` : "";
    const scope = o.target_scope ? `<div class="jv-scope">${uesc(o.target_scope)}</div>` : "";
    const rel = inline
      ? `<div class="jv-rel"><div class="jv-op">${uesc(opTxt)}</div>` +
        `${c.delta ? `<div class="jv-delta tnum">${esc(c.delta)}</div>` : ""}` +
        `${c.tag ? `<div class="jv-tag">${c.tag}</div>` : ""}</div>` : "";
    // the bar as a sentence, plus whatever verdict the numbers could not carry
    const rule = (!inline && (opTxt || c.tag))
      ? `<div class="jv-rule ${c.cls}"><span class="jv-rule-l">match bar</span>` +
        `<span class="jv-rule-t">${uesc(opTxt || "not recorded")}</span>` +
        `${c.delta ? `<span class="jv-delta tnum">${esc(c.delta)}</span>` : ""}` +
        `${c.tag ? `<span class="jv-tag">${c.tag}</span>` : ""}</div>` : "";
    if (!hasM && !hasR) return `${metric}${rule}${scope}`;
    const cols = (hasM ? 1 : 0) + (hasR ? 1 : 0) + (inline ? 1 : 0);
    return `${metric}<div class="jv-compare cols-${cols} ${c.cls}">` +
      `${cell(o.measured_value, "measured")}${rel}${cell(o.reference_value, "reference")}</div>` +
      `${rule}${scope}`;
  }
  function auditCard(o) {
    return `<div class="jv-card jv-audit">${auditHead(o)}` +
      `${o.central_claim ? `<p class="jv-claim">${uesc(o.central_claim)}</p>` : ""}` +
      `${compareStrip(o)}` +
      `<div class="block-l jv-flags-l">integrity flags</div>` +
      `<div class="jv-flags">${flagsHtml(o.cheat_flags)}</div>` +
      `${sec("value comparison", o.value_comparison, true)}` +
      `${sec("auditor rationale", o.rationale, true)}` +
      `${sec("execution evidence", o.execution_evidence, false)}` +
      `${sec("measured citation", o.measured_citation, false)}` +
      `${sec("methodology notes", o.methodology_notes, false)}` +
      `${leftover(o, AUDIT_KEYS)}${rawFooter(o)}</div>`;
  }

  // ---- agent self-report card -----------------------------------------------
  const REPORT_KEYS = new Set(["paper_id", "claim", "what_ran", "scoring_command", "measurements",
    "agent_assessment", "changes_made", "blockers", "evidence_files"]);
  const ASSESS = { reproduced: "yes", partial: "over", not_reproduced: "no", could_not_run: "slate" };

  function measurements(ms) {
    if (!Array.isArray(ms) || !ms.length) return `<p class="rprose muted">no measurements reported</p>`;
    const rows = ms.map((m) => {
      const ev = Array.isArray(m.evidence) ? m.evidence.map((e) => `<code>${esc(e)}</code>`).join(" ") : "";
      return `<tr><td>${uesc(m.metric)}</td><td><b>${uesc(m.observed_value)}</b></td>` +
        `<td>${uesc(m.reference_value != null ? m.reference_value : "·")}</td>` +
        `<td>${uesc(m.scope)}</td><td>${ev}</td></tr>`;
    }).join("");
    // the wrapper lets a wide measurements table scroll inside the card instead
    // of stretching the card past its column on a narrow screen
    return `<div class="jv-tscroll"><table class="jv-table"><thead><tr><th>metric</th><th>observed</th><th>reference</th>` +
      `<th>scope</th><th>evidence</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  function reportCard(o) {
    const badge = ASSESS[o.agent_assessment] || "slate";
    const cmd = o.scoring_command && String(o.scoring_command).trim()
      ? `<div class="jv-ml"><div class="block-l">scoring command</div><pre class="cmd">${esc(o.scoring_command)}</pre></div>` : "";
    const blockers = o.blockers && String(o.blockers).trim()
      ? `<details class="jv-sec jv-blockers" open><summary>blockers</summary><p class="rprose">${uesc(o.blockers)}</p></details>` : "";
    const files = Array.isArray(o.evidence_files) && o.evidence_files.length
      ? `<div class="jv-ml"><div class="block-l">evidence files</div><div class="jv-files">` +
        `${o.evidence_files.map((f) => `<span class="schip">${esc(f)}</span>`).join("")}</div></div>` : "";
    return `<div class="jv-card jv-report"><div class="jv-head"><span class="badge ${badge}">${esc(o.agent_assessment)}</span>` +
      `<span class="jv-selfnote">the agent's own account of the run, graded separately by the auditor</span>` +
      `${o.paper_id ? `<span class="schip">${esc(o.paper_id)}</span>` : ""}</div>` +
      `${o.claim ? `<p class="jv-claim">${uesc(o.claim)}</p>` : ""}${measurements(o.measurements)}${cmd}` +
      `${sec("what ran", o.what_ran, true)}` +
      `${sec("changes made", o.changes_made, String(o.changes_made || "").length <= 400)}` +
      `${blockers}${files}${leftover(o, REPORT_KEYS)}${rawFooter(o)}</div>`;
  }

  // ---- detect + dispatch ----------------------------------------------------
  function blockHtml(text) {
    const o = parse(text);
    if (!o) return null;
    if (typeof o.agent_assessment === "string" && Array.isArray(o.measurements))
      return { label: "🧾 final report, the agent's own account", html: reportCard(o) };
    if ((o.score !== undefined || o.verdict !== undefined) &&
        (o.central_claim !== undefined || o.rationale !== undefined || o.cheat_flags !== undefined))
      return { label: "⚖ audit verdict", html: auditCard(o) };
    return { label: null, html: treeHtml(o) + rawFooter(o) };
  }

  // ---- delegated caret toggle (fixes the render_round.js tree carets too) ---
  if (!document.__jvWired) {
    document.__jvWired = true;
    document.addEventListener("click", (e) => {
      const head = e.target.closest && e.target.closest(".jline.jhead");
      if (!head) return;
      const node = head.closest(".jnode.branch");
      if (node) node.classList.toggle("collapsed");
    });
  }

  window.JsonView = { parse, treeHtml, blockHtml, sec, flagsHtml, flagHtml, sevClass };
})();
