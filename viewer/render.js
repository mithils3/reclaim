/* render.js: shared rendering core + the run transcript shell. Turns the
   normalized Run/Round/Call shape into DOM. Pure helpers (esc/el/fmt*) live here
   and are exported on window.RENDER; round-card rendering is in render_round.js
   and the small card helpers in runcard.js. The run detail is an instrument panel
   (tile strip + horizontal burn trace) over a two-column strip chart: a vertical
   burn-trace rail (trace.js/strip.js) beside collapsible timeline round cards. */
"use strict";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// Some captured payloads were JSON-encoded twice, so a non-ASCII character
// survives the parse as the six literal characters that name it: the auditor's
// prose then prints a backslash-u escape where a π, a ± or an arrow belongs.
// uni() decodes those back to the character they name. ONLY codepoints at
// U+00A0 and above are decoded, so an escape naming "<" can never turn escaped
// text back into markup and uni() is safe to run either side of esc().
function uni(s) {
  const t = String(s ?? "");
  if (t.indexOf("\\u") < 0) return t;
  return t.replace(/\\u([0-9a-fA-F]{4})/g, (m, hex) => {
    const code = parseInt(hex, 16);
    return code >= 0xa0 ? String.fromCharCode(code) : m;
  });
}
const uesc = (s) => esc(uni(s));
const num = (v) => (v == null ? null : (Math.round(v * 10000) / 10000));

// H100·h as a readable Hh MMm / MMm SSs / SSs (compute, so "minutes" = H100·minutes)
function fmtHM(h) {
  if (h == null || isNaN(h)) return "·";
  const neg = h < 0, sec = Math.round(Math.abs(h) * 3600);
  const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  const s = hh > 0 ? `${hh}h ${String(mm).padStart(2, "0")}m`
    : mm > 0 ? `${mm}m ${String(ss).padStart(2, "0")}s` : `${ss}s`;
  return (neg ? "−" : "") + s;
}
// wall duration in seconds -> "3d 04h" / "8h 40m" / "12m 30s"
function fmtDur(s) {
  if (s == null || isNaN(s)) return "·";
  const t = Math.max(0, Math.round(s));
  const d = Math.floor(t / 86400), h = Math.floor((t % 86400) / 3600);
  const m = Math.floor((t % 3600) / 60), ss = t % 60;
  if (d > 0) return `${d}d ${String(h).padStart(2, "0")}h`;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(ss).padStart(2, "0")}s`;
  return `${ss}s`;
}
const fmtTok = (n) => n == null ? "·" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(n);
const fmtTime = () => "";

// ---- how a run ended -------------------------------------------------------
// The index carries the reader-facing exit_label. A round event still carries the
// raw word, so the same map turns it into the same label on the final card.
const EXIT_LABEL = {
  natural: "Finished", completed: "Finished",
  budget_exhausted: "Budget exhausted",
  context_budget: "Context limit",
  round_limit: "Round limit",
  wall_clock: "Time limit", timeout: "Time limit",
  error: "Ended with error",
};
const exitLabel = (raw) => (raw ? (EXIT_LABEL[String(raw)] || "Ended") : null);
const GOOD_EXIT = (e) => e === "natural" || e === "completed";

// ---- paper joins (carried on the run row by data.js) ------------------------
const claimOf = (run) => (run && run.claim) || null;
function predictedOf(run) {
  if (!run) return null;
  const p = num(run.predicted_h100);
  return (p != null && p > 0) ? p : num(run.budget_h100);
}
const V = () => window.Verdict;

// spent / total / predicted from a run row (single source for the tiles and bars)
function fuelNums(run) {
  const total = num(run.budget_h100);
  const spent = num(run.spent_h100);
  const predicted = predictedOf(run);
  return { total, spent, predicted };
}

// ---- run header → instrument panel (tile strip) ----------------------------
function itile(label, value, sub, cls) {
  return `<div class="itile ${cls || ""}"><div class="it-l">${label}</div><div class="it-v tnum">${value}</div>${sub ? `<div class="it-sub">${sub}</div>` : ""}</div>`;
}
function runTilesHtml(run, extra) {
  const { total, spent } = fuelNums(run);
  // a run that spends a sliver rounds to "0% of budget", which reads as nothing
  // spent; the same "<1" the runs table uses keeps the tile honest
  const share = (total && spent != null) ? (spent / total) * 100 : null;
  const pct = share == null ? ""
    : `${share > 0 && share < 1 ? "<1" : Math.round(share)}% of budget`;
  const rounds = (extra && extra.rounds != null) ? String(extra.rounds) : (run.rounds != null ? String(run.rounds) : "·");
  const tok = run.tokens || {};
  // the paper's own compute estimate, never the budget fallback: the tile is
  // omitted when the record carries no estimate for this paper
  const est = num(run.predicted_h100);
  return `<div class="itiles">` +
    itile("SPENT", fmtHM(spent), pct, "big") +
    itile("BUDGET", fmtHM(total), "H100·h", "") +
    ((est != null && est > 0) ? itile("ESTIMATED NEED", fmtHM(est), "paper&#39;s estimate", "") : "") +
    itile("ROUNDS", rounds, run.tool_calls != null ? `${run.tool_calls} tool calls` : "", "") +
    itile("TOKENS", fmtTok(tok.total), tok.cached != null ? `${fmtTok(tok.cached)} cached` : "", "") +
    itile("DURATION", fmtDur(run.duration_s), "wall clock", "") +
    `</div>`;
}

function runHeaderHtml(run, extra) {
  const fam = V() ? V().ofRun(run) : "done";
  const word = V() ? V().word(run) : "";
  const claim = claimOf(run);
  const target = (typeof run.target === "string" && run.target.trim()) ? run.target.trim() : null;
  const a = run.audit || {};
  const exit = run.exit_label
    ? `<span class="badge ${run.exit_label === "Finished" ? "yes" : "over"}">${esc(run.exit_label)}</span>` : "";
  const score = a.score != null
    ? `<span class="an-score ${a.score >= 8 ? "yes" : a.score >= 6 ? "over" : a.score <= 0 ? "no" : "slate"}">${esc(a.score)}<span>/10</span></span>` : "";
  const mode = (run.mode && window.Modes) ? window.Modes.chip(run.mode) : "";
  const tierName = window.Data ? window.Data.tierName(run.tier) : run.tier;
  const tierCls = { run: "tier-run", retrain: "tier-retrain", reimplement: "tier-reimplement" }[run.tier] || "slate";
  const tierWhat = window.Data ? (window.Data.tier(run.tier).what || "") : "";
  const tier = run.tier ? `<span class="badge ${tierCls}" title="${esc(tierWhat)}">${esc(tierName)}</span>` : "";
  const model = run.model_name ? `<span class="schip">${esc(run.model_name)}</span>` : "";
  const p = run.paper || {};
  const lk = (p.paper_url || p.code_url)
    ? `<span class="rt-links">${p.paper_url ? `<a href="${esc(p.paper_url)}" target="_blank" rel="noopener">paper↗</a>` : ""}${p.code_url ? `<a href="${esc(p.code_url)}" target="_blank" rel="noopener">code↗</a>` : ""}</span>` : "";
  const arx = run.arxiv_id
    ? `<a class="pid" href="https://arxiv.org/abs/${esc(run.arxiv_id)}" target="_blank" rel="noopener">${esc(run.arxiv_id)} ↗</a>` : "";
  return `<div class="rt-head">${V() ? V().stamp(fam, word ? String(word).replace(/_/g, " ").toUpperCase() : null) : ""}
      ${tier}${model}${score}${mode}${exit}${lk}</div>
    <h2 class="rt-claim">${uesc(claim || run.arxiv_id || "transcript")}</h2>
    ${target ? `<div class="rt-target"><span class="rt-target-l">TARGET</span><span class="rt-target-v">${uesc(target)}</span></div>` : ""}
    <div class="rt-meta">${arx}</div>
    ${runTilesHtml(run, extra)}`;
}
const topHtml = (run, extra) => runHeaderHtml(run, extra);

// ---- horizontal burn trace (lives OUTSIDE .run-top so patches don't wipe it) --
// The chart alone reads as a flat line whenever a run finishes well under the
// paper's estimate, which is the common case, so the panel carries the units, a
// key, the round range and a took-vs-predicted ledger caption around it.
// the viewBox aspect follows the column the panel lands in: a 3:25 strip on a
// wide screen, a squarer box on a phone, so the plot never collapses to a band
// too short to carry its own value labels
const trBox = () => (window.innerWidth < 760
  ? { w: 380, h: 152, pad: 8 }
  : { w: 720, h: 86, pad: 8 });

function traceSvg(run, rounds, TR) {
  if (!window.Trace || !rounds || !rounds.length) return "";
  const predicted = predictedOf(run);
  const family = V() ? V().ofRun(run) : "done";
  return window.Trace.draw(rounds, { orientation: "h", vbW: TR.w, vbH: TR.h, pad: TR.pad,
    fill: true, band: true, markers: true, interest: true, axis: true, spike: run.__spike, predicted, family,
    density: (TR.w - 2 * TR.pad) / Math.max(rounds.length - 1, 1),
    idOf: (p) => { const r = rounds[p.i]; return r && r.round_index != null ? r.round_index : p.i; },
    label: `cumulative compute over ${rounds.length} rounds` });
}

function tracePanelHtml(run, rounds) {
  if (!window.Trace || !rounds || !rounds.length) return "";
  const TR = trBox();
  const cum = window.Trace.cumulative(rounds);
  const total = cum.total;
  const pred = predictedOf(run);
  const spent = num(run.spent_h100);
  const took = spent != null ? spent : total;
  // predictedOf falls back to the metered budget when the record carries no
  // estimate for the paper, so the ceiling is named for what it actually is
  const est = num(run.predicted_h100);
  const fromPaper = est != null && est > 0;
  const ceil = fromPaper ? "the paper&#39;s estimate" : "the metered budget";
  const noun = fromPaper ? "estimate" : "budget";
  // Nothing metered means there is no curve to draw: plotting it gives a flat
  // line on the baseline under an empty box. The panel drops the plot and keeps
  // the head and one line; the rail and the round cards still carry the
  // transcript. The ceiling is still named when the record carries one.
  if (!(total > 0)) {
    const why = (pred != null && pred > 0)
      ? `no compute was metered against ${ceil} of ${fmtHM(pred)}`
      : "no compute was metered on this transcript";
    return `<figure class="run-trace tr-flat">
      <figcaption class="tr-head"><span class="plate">compute burn</span>
        <span class="tr-sub">${why}</span>
        <span class="tr-sub tr-sub-2">${rounds.length} round${rounds.length === 1 ? "" : "s"}, in order below</span></figcaption>
    </figure>`;
  }
  const faults = cum.pts.some((p) => p.fault);
  const over = pred != null && total > pred;
  // the plot area is annotated rather than left blank: the axis top, the dashed
  // ceiling and the curve's own end all print their value at their own height,
  // so the headroom under a paper's estimate reads as a measured quantity.
  // scaleOf is the same call the SVG makes, so the labels and the curve can
  // never sit on two different axes: under a fifth of the ceiling it scales to
  // the spend and the ceiling leaves the plot for the key.
  const sc = window.Trace.scaleOf(total, pred);
  const yMax = sc.yMax, offScale = sc.off;
  const aH = TR.h - 2 * TR.pad;
  const topOf = (v) => ((TR.pad + aH - (v / yMax) * aH) / TR.h) * 100;
  const predTop = (pred != null && pred > 0 && !offScale) ? topOf(pred) : null;
  // the curve's own label is suppressed when it would collide with the ceiling
  const endTop = total > 0 ? topOf(total) : null;
  const showEnd = endTop != null && (predTop == null || Math.abs(endTop - predTop) > 13);
  const share = (offScale && pred > 0)
    ? (total / pred >= 0.01 ? `${Math.round((total / pred) * 100)}%` : "under 1%") : null;
  const key = [
    `<span class="lg"><i class="tr-sw tr-sw-line"></i>cumulative spend</span>`,
    (pred != null && !offScale) ? `<span class="lg"><i class="tr-sw tr-sw-band"></i>${ceil} and the allowance under it</span>` : "",
    offScale ? `<span class="lg tr-lg-off"><i class="tr-sw tr-sw-off"></i>${ceil}, ${fmtHM(pred)}, is off this scale` +
      `${share ? ` (${share} of it spent)` : ""}</span>` : "",
    over ? `<span class="lg"><i class="tr-sw tr-sw-over"></i>over ${ceil}</span>` : "",
    faults ? `<span class="lg"><i class="tr-sw tr-sw-fault"></i>failed call</span>` : "",
  ].join("");
  const ratio = (pred != null && pred > 0 && took != null) ? took / pred : null;
  const pct = ratio != null ? Math.round(ratio * 100) : null;
  // a spend under half a percent rounds to "100% of the budget unspent", which
  // claims more precision than the number carries. The ledger's own "<1%" chip
  // already states it, so the second clause drops.
  const second = ratio == null ? ""
    : ratio >= 2 ? `${(Math.round(ratio * 10) / 10)}× ${noun === "estimate" ? "the estimate" : "the budget"}`
      : pct >= 100 ? `${pct - 100}% over ${noun === "estimate" ? "the estimate" : "the budget"}`
        : pct <= 0 ? ""
          : `${100 - pct}% of the ${noun} unspent`;
  const foot = (pred != null && pred > 0)
    ? `${window.Trace.ledger(took, pred, { pct: ratio == null || ratio < 10 })}` +
      `<span class="tr-cap">took <b class="tnum">${fmtHM(took)}</b> of <b class="tnum">${fmtHM(pred)}</b> ${fromPaper ? "predicted" : "budgeted"}` +
      `${second ? `<span class="tr-cap-2">${second}</span>` : ""}</span>`
    : `<span class="tr-cap">took <b class="tnum">${fmtHM(took)}</b>` +
      `<span class="tr-cap-2">no ceiling recorded for this paper</span></span>`;
  const a = rounds[0], z = rounds[rounds.length - 1];
  const rlab = (r, fb) => (r && r.round_index != null ? `round ${esc(r.round_index)}` : fb);
  const axisNote = offScale ? ` <span class="tr-sub-2">axis scaled to the spend</span>` : "";
  return `<figure class="run-trace">
    <figcaption class="tr-head"><span class="plate">compute burn</span>
      <span class="tr-sub">cumulative H100&#183;h across ${rounds.length} round${rounds.length === 1 ? "" : "s"}${axisNote}</span>
      <span class="chart-legend tr-key">${key}</span></figcaption>
    <div class="tr-plot">${traceSvg(run, rounds, TR)}
      <span class="tr-y tr-y-max tnum">${fmtHM(yMax)}</span>
      <span class="tr-y tr-y-zero tnum">0</span>
      ${predTop != null ? `<span class="tr-pred-l tnum" style="top:${predTop.toFixed(2)}%">${fmtHM(pred)} ${fromPaper ? "predicted" : "budget"}</span>` : ""}
      ${showEnd ? `<span class="tr-end-l tnum ${over ? "over" : ""}" style="top:${endTop.toFixed(2)}%">${fmtHM(total)}</span>` : ""}
    </div>
    <div class="tr-ax"><span>${rlab(a, "first round")}</span><span>${rlab(z, "last round")}</span></div>
    <div class="tr-foot">${foot}</div>
  </figure>`;
}

// ---- transcript shell ------------------------------------------------------
function setSpike(run, rounds) { if (window.Trace) run.__spike = window.Trace.cumulative(rounds || []).maxDelta; }
// default collapse on a full render: everything collapsed except the last two + finals
const collapseDefault = (round, i, n) => round.kind !== "final" && i < n - 2;
const roundsCtlHtml = (nGroups) =>
  `<div class="rounds-ctl">${nGroups ? `<span class="rc-note">${nGroups} routine stretch${nGroups === 1 ? "" : "es"} folded</span>` : ""}` +
  `<button class="link" type="button" data-act="expand">expand all</button>` +
  `<button class="link" type="button" data-act="collapse">collapse all</button></div>`;
function wireRounds(rootEl) {
  const roundsEl = rootEl.querySelector(".rounds");
  if (!roundsEl || roundsEl.__wired) return;
  roundsEl.__wired = true;
  roundsEl.addEventListener("click", (e) => {
    const ctl = e.target.closest(".rounds-ctl [data-act]");
    if (ctl) {
      const expand = ctl.dataset.act === "expand";
      roundsEl.querySelectorAll(".rgroup").forEach((g) => {
        g.classList.toggle("collapsed", !expand);
        const gh = g.querySelector(".rgroup-h");
        if (gh) gh.setAttribute("aria-expanded", String(expand));
      });
      roundsEl.querySelectorAll(".rcard").forEach((c) => {
        c.classList.toggle("collapsed", !expand);
        const ch = c.querySelector(".rcard-h");
        if (ch) ch.setAttribute("aria-expanded", String(expand));
      });
      return;
    }
    const g = e.target.closest(".rgroup-h");
    if (g && roundsEl.contains(g)) {
      const grp = g.closest(".rgroup");
      const c = grp.classList.toggle("collapsed");
      g.setAttribute("aria-expanded", String(!c));
      return;
    }
    const h = e.target.closest(".rcard-h");
    if (h && roundsEl.contains(h)) { const card = h.closest(".rcard"); const c = card.classList.toggle("collapsed"); h.setAttribute("aria-expanded", String(!c)); }
  });
}

// ---- routine-round folding -------------------------------------------------
// A 94-round transcript is mostly one tool called twice per round. Printing 94
// near-identical one-line cards buries the rounds that carry the story, so a run
// of three or more consecutive rounds that do the same routine thing folds into
// one summary row that opens on click. A round escapes the fold when it fails a
// call, gets cut off, costs a real share of the run's costliest round, carries a
// substantial amount of prose, is the final round, or is one of the last two.
const NOTE_CHARS = 3000, GROUP_MIN = 3;
function tallyKey(calls) {
  const m = new Map();
  for (const c of (calls || [])) { const t = c.tool_name || "?"; m.set(t, (m.get(t) || 0) + 1); }
  return [...m].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([t, k]) => `${t}×${k}`).join(" · ");
}
function notable(round, pt, run, i, n) {
  if (round.kind === "final" || i >= n - 2) return true;
  if (round.finish_reason === "length") return true;
  if (pt && pt.fault) return true;
  if (((round.reasoning || "") + (round.content || "")).trim().length > NOTE_CHARS) return true;
  const spike = run && run.__spike > 0 ? run.__spike : 0;
  if (spike > 0 && pt && pt.delta >= spike * 0.5) return true;
  return false;
}
// [{ group:false, round, i } | { group:true, rounds }]
function foldRounds(rounds, run) {
  const pts = (window.Trace ? window.Trace.cumulative(rounds).pts : []);
  const n = rounds.length, out = [];
  let buf = [];
  const flushBuf = () => {
    if (!buf.length) return;
    if (buf.length >= GROUP_MIN) out.push({ group: true, rounds: buf.map((b) => b.r) });
    else buf.forEach((b) => out.push({ group: false, round: b.r, i: b.i }));
    buf = [];
  };
  rounds.forEach((r, i) => {
    if (notable(r, pts[i], run, i, n)) { flushBuf(); out.push({ group: false, round: r, i }); return; }
    buf.push({ r, i });
  });
  flushBuf();
  return out;
}
// the summary row states what the folded rounds all did, so folding hides
// repetition and never hides a distinction
function groupHtml(item, arxiv, run) {
  const rs = item.rounds, a = rs[0], z = rs[rs.length - 1];
  let cost = 0, any = false, calls = 0;
  const keys = new Set(), tools = new Map();
  for (const r of rs) {
    keys.add(tallyKey(r.calls));
    for (const c of (r.calls || [])) {
      calls++;
      const t = c.tool_name || "?";
      tools.set(t, (tools.get(t) || 0) + 1);
      if (typeof c.cost_h100 === "number") { cost += c.cost_h100; any = true; }
    }
  }
  const uniform = keys.size === 1 ? [...keys][0] : null;
  const what = calls === 0 ? "no tool calls"
    : uniform ? `${esc(uniform)} each`
      : `${calls} tool calls · ${esc([...tools.keys()].slice(0, 3).join(", "))}`;
  const span = `rounds ${esc(a.round_index)}–${esc(z.round_index)}`;
  const cst = any ? `<span class="rg-cost tnum">+${fmtHM(Math.round(cost * 1e6) / 1e6)}</span>` : "";
  const cards = rs.map((r) => window.RENDER.roundCardHtml(r, arxiv, run, true)).join("");
  return `<div class="rgroup collapsed" data-round="${esc(a.round_index ?? "")}">
    <button class="rgroup-h" type="button" aria-expanded="false"
      title="open these ${rs.length} routine rounds"><span class="rg-idx">${span}</span>
      <span class="rg-n">${rs.length} routine rounds</span><span class="rg-what">${what}</span>${cst}</button>
    <div class="rgroup-body">${cards}</div></div>`;
}

// open a round by its index and return the card, unfolding the group it sits in
function revealRound(rootEl, idx) {
  if (!rootEl || idx == null) return null;
  const card = rootEl.querySelector(`.rcard[data-round="${CSS.escape(String(idx))}"]`);
  if (!card) return null;
  const g = card.closest(".rgroup");
  if (g && g.classList.contains("collapsed")) {
    g.classList.remove("collapsed");
    const gh = g.querySelector(".rgroup-h");
    if (gh) gh.setAttribute("aria-expanded", "true");
  }
  card.classList.remove("collapsed");
  const h = card.querySelector(".rcard-h");
  if (h) h.setAttribute("aria-expanded", "true");
  return card;
}
// opts.head === false renders the trace + rail + rounds only, so a caller can
// place its own header and the audit / dissection cards above the transcript.
function renderRun(rootEl, run, rounds, opts) {
  const RR = window.RENDER;
  opts = opts || {};
  rounds = rounds || [];
  setSpike(run, rounds);
  const n = rounds.length;
  const head = opts.head === false ? "" : `<div class="run-top">${topHtml(run, { rounds: n })}</div>`;
  // no rounds: say so, rather than shipping an empty rail beside an empty column
  if (!n) {
    rootEl.innerHTML = `<div class="run-detail">${head}` +
      `<div class="panel-card rd-norounds"><div class="pc-head"><span class="plate">transcript</span></div>` +
      `<p class="rprose muted">No rounds were recorded for this run.</p></div></div>`;
    return;
  }
  const items = foldRounds(rounds, run);
  const nGroups = items.filter((it) => it.group).length;
  const body = items.map((it) => (it.group
    ? groupHtml(it, run.arxiv_id, run)
    : RR.roundCardHtml(it.round, run.arxiv_id, run, collapseDefault(it.round, it.i, n)))).join("");
  rootEl.innerHTML = `<div class="run-detail">
    ${head}
    ${tracePanelHtml(run, rounds)}
    <div class="run-body"><div class="strip-rail"></div>
      <div class="rounds">${roundsCtlHtml(nGroups)}${body}</div></div>
  </div>`;
  wireRounds(rootEl);
  if (window.Strip) window.Strip.mount(rootEl, run, rounds);
}

window.RENDER = { esc, uni, uesc, num, fmtHM, fmtDur, fmtTok, fmtTime, claimOf, predictedOf,
  GOOD_EXIT, exitLabel, fuelNums, renderRun, topHtml, runTilesHtml, revealRound };
