/* runs.js: the runs table. Every published run as one sortable row: the claim it
   was asked to reproduce, the paper id, the agent, the tier, the audit score and
   verdict, the primary failure mode, rounds, spend against budget and tokens.
   The verdict, mode and text filters sit on top of the global Agent and Tier
   selects, and the resulting order is what the run page's prev and next follow.

   Row anatomy is fixed: the claim is clamped to two lines and carries the full
   text on title, every other cell is a single line, so 275 rows read as one
   even grid rather than a ragged column of paragraphs. */
"use strict";

(function () {
  const R = window.RENDER, esc = R.esc, V = () => window.Verdict;
  const TIER_CLS = { run: "tier-run", retrain: "tier-retrain", reimplement: "tier-reimplement" };

  const COLS = [
    { k: "claim", l: "claim" },
    { k: "arxiv_id", l: "paper" },
    { k: "model", l: "agent" },
    { k: "tier", l: "tier" },
    { k: "score", l: "score", num: true },
    { k: "verdict", l: "verdict" },
    { k: "mode", l: "failure mode" },
    { k: "rounds", l: "rounds", num: true },
    { k: "spent", l: "spent / budget", num: true },
    { k: "tokens", l: "tokens", num: true },
  ];

  const val = (r, k) => {
    switch (k) {
      case "score": return r.audit_score == null ? -1 : r.audit_score;
      case "verdict": return r.audit_verdict || "";
      case "mode": return window.Modes.rank(r.mode);
      case "spent": return r.spent_h100 == null ? -1 : r.spent_h100;
      case "tokens": return (r.tokens && r.tokens.total) || 0;
      case "model": return r.model_name || r.model || "";
      case "tier": return window.Data.tierRank(r.tier);
      default: return r[k] == null ? "" : r[k];
    }
  };
  const NUMERIC = { score: 1, rounds: 1, spent: 1, tokens: 1, tier: 1, mode: 1 };

  // a run that did not end on its own says so beside its round count
  function exitMark(r) {
    if (!r.exit_label || r.exit_label === "Finished") return "";
    return `<span class="rt-exit" title="${esc(r.exit_label)}" aria-label="${esc(r.exit_label)}">▣</span>`;
  }

  const Runs = {
    sortKey: "score", sortDir: -1,

    root() { return document.querySelector("#runs-root"); },

    ordered() {
      const S = window.State, q = (S.q || "").trim().toLowerCase();
      const rows = window.Data.runs.filter((r) =>
        (S.model === "all" || r.model === S.model) &&
        (S.tier === "all" || r.tier === S.tier) &&
        (S.verdict === "all" || r.audit_verdict === S.verdict) &&
        (S.mode === "all" || r.mode === S.mode) &&
        (!q || `${r.claim || ""} ${r.arxiv_id} ${r.model_name || ""} ${r.mode_name || ""} ${r.audit_verdict || ""}`.toLowerCase().includes(q)));
      const k = this.sortKey, dir = this.sortDir;
      return rows.sort((a, b) => {
        const x = val(a, k), y = val(b, k);
        const c = NUMERIC[k] ? (x - y) * dir : String(x).localeCompare(String(y)) * dir;
        return c || String(a.id).localeCompare(String(b.id));
      });
    },

    rowHtml(r) {
      const fam = V().ofRun(r);
      const claim = r.claim || r.arxiv_id;
      const score = r.audit_score == null ? "·"
        : `<span class="an-score ${r.audit_score >= 8 ? "yes" : r.audit_score >= 6 ? "over" : r.audit_score <= 0 ? "no" : "slate"}">${esc(r.audit_score)}<span>/10</span></span>`;
      const tok = (r.tokens && r.tokens.total) || null;
      const mode = window.Modes.name(r.mode);
      return `<tr data-run="${esc(r.id)}" tabindex="0">
        <td class="pl-claim-cell c-claim"><span class="vg vd ${fam}">${V().meta(fam).glyph}</span><span class="pcell-claim" title="${esc(claim)}">${esc(claim)}</span></td>
        <td class="s-rid c-paper">${esc(r.arxiv_id)}</td>
        <td class="s-model c-agent">${esc(r.model_name || r.model)}</td>
        <td class="c-tier"><span class="badge ${TIER_CLS[r.tier] || "slate"}">${esc(window.Data.tierName(r.tier))}</span></td>
        <td class="num c-score">${score}</td>
        <td class="c-verdict">${V().inline(fam, V().word(r))}</td>
        <td class="mode-cell c-mode" title="${esc(mode)}">${window.Modes.chip(r.mode)}</td>
        <td class="num tnum c-rounds" data-l="rounds">${exitMark(r)}${r.rounds == null ? "·" : esc(r.rounds)}</td>
        <td class="fuel-cell c-fuel">${R.microFuelHtml(r)}</td>
        <td class="num tnum c-tokens" data-l="tokens">${R.fmtTok(tok)}</td>
      </tr>`;
    },

    bodyHtml(rows) {
      if (!rows.length) return `<tr><td colspan="${COLS.length}" class="empty small">No runs match these filters.` +
        `<span>Clear the search, or widen the verdict and the failure mode.</span></td></tr>`;
      return rows.map((r) => this.rowHtml(r)).join("");
    },

    headHtml() {
      return COLS.map((c) => {
        const on = this.sortKey === c.k;
        const dir = this.sortDir < 0 ? "descending" : "ascending";
        return `<th class="${c.num ? "num" : ""} ${on ? "sorted" : ""}" data-k="${c.k}" tabindex="0" role="button"` +
          ` aria-sort="${on ? dir : "none"}" title="sort by ${esc(c.l)}">${esc(c.l)}` +
          `<span class="th-dir">${on ? (this.sortDir < 0 ? "▼" : "▲") : ""}</span></th>`;
      }).join("");
    },

    filtersHtml() {
      const S = window.State;
      const verdicts = [...new Set(window.Data.runs.map((r) => r.audit_verdict).filter(Boolean))].sort();
      const modes = window.Modes.all.filter((m) => window.Data.runs.some((r) => r.mode === m.key));
      const opt = (v, l, cur) => `<option value="${esc(v)}"${v === cur ? " selected" : ""}>${esc(l)}</option>`;
      return `<div class="ps-filters">
        <select id="rn-verdict" aria-label="verdict">${[opt("all", "all verdicts", S.verdict)].concat(verdicts.map((v) => opt(v, v.replace(/_/g, " "), S.verdict))).join("")}</select>
        <select id="rn-mode" aria-label="failure mode">${[opt("all", "all failure modes", S.mode)].concat(modes.map((m) => opt(m.key, m.name, S.mode))).join("")}</select>
        <input id="rn-q" class="s-search" type="search" placeholder="search runs…" aria-label="search runs" value="${esc(S.q)}" />
      </div>`;
    },

    countHtml(n) {
      const all = window.Data.runs.length;
      return n === all ? `${all} runs` : `${n} of ${all} runs`;
    },

    render() {
      const host = this.root();
      if (!host) return;
      host.classList.add("wide-list");   // ten columns need more than the prose measure
      const rows = this.ordered();
      host.innerHTML = `
        <div class="ov-head">
          <div><h1>Runs</h1><div class="ov-sub">One row per reproduction run. Open a row for the transcript, the audit and the dissection.</div></div>
        </div>
        <div class="ps-head"><span class="plate">runs</span><span class="ps-count" id="rn-count">${this.countHtml(rows.length)}</span>${this.filtersHtml()}</div>
        <div class="tscroll"><table class="stats-table runs-table"><thead><tr>${this.headHtml()}</tr></thead>
          <tbody id="rn-body">${this.bodyHtml(rows)}</tbody></table></div>`;
      this.wire();
    },

    renderBody() {
      const rows = this.ordered();
      const body = document.querySelector("#rn-body");
      const count = document.querySelector("#rn-count");
      if (body) body.innerHTML = this.bodyHtml(rows);
      if (count) count.textContent = this.countHtml(rows.length);
      this.wireRows();
      history.replaceState(null, "", window.runsHash());
    },

    wireRows() {
      document.querySelectorAll("#rn-body tr[data-run]").forEach((tr) => {
        const open = () => window.go("#/run/" + encodeURIComponent(tr.dataset.run));
        tr.onclick = open;
        tr.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); open(); } };
      });
    },

    wire() {
      const host = this.root();
      host.querySelectorAll("thead th[data-k]").forEach((th) => {
        const sort = () => {
          const k = th.dataset.k;
          if (this.sortKey === k) this.sortDir *= -1;
          else { this.sortKey = k; this.sortDir = COLS.find((c) => c.k === k).num ? -1 : 1; }
          this.render();
        };
        th.onclick = sort;
        th.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } };
      });
      const v = host.querySelector("#rn-verdict");
      if (v) v.onchange = () => { window.State.verdict = v.value; this.renderBody(); };
      const m = host.querySelector("#rn-mode");
      if (m) m.onchange = () => { window.State.mode = m.value; this.renderBody(); };
      const q = host.querySelector("#rn-q");
      if (q) q.oninput = () => { window.State.q = q.value; this.renderBody(); };
      this.wireRows();
    },
  };

  window.Runs = Runs;
})();
