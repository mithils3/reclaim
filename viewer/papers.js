/* papers.js: the papers grid and one paper's page. The grid puts every target
   paper on a row and every agent in a column, so a row reads as how the field
   fared on that claim and a column reads as one agent across the whole set.
   A cell is the audit score, coloured by its verdict family, or blank where that
   agent has no run on that paper. The row closes with the mean of the scored
   cells, which is what the column sort ranks on. Opening a row lists that
   paper's runs under an instrument strip for the paper itself. */
"use strict";

(function () {
  const R = window.RENDER, esc = R.esc, V = () => window.Verdict;
  const TIER_CLS = { run: "tier-run", retrain: "tier-retrain", reimplement: "tier-reimplement" };

  // one agent's score on one paper. The agent name rides on data-l so the cell can
  // name itself once the head row is gone on a phone.
  function cellHtml(run, dim, model) {
    const l = esc((model && model.name) || "");
    if (!run) return `<td class="pm-cell c-m empty" data-l="${l}">·</td>`;
    const fam = V().ofRun(run);
    const s = run.audit_score;
    return `<td class="pm-cell c-m ${dim ? "dim" : ""}" data-l="${l}"><button class="pm-chip ${fam}" data-run="${esc(run.id)}"
      title="${esc(run.model_name)} · ${esc(V().word(run))} · ${esc(run.mode_name || "")}">${s == null ? "·" : esc(s)}</button></td>`;
  }

  // a run that did not end on its own says so beside its round count
  function exitMark(r) {
    if (!r.exit_label || r.exit_label === "Finished") return "";
    return `<span class="rt-exit" title="${esc(r.exit_label)}" aria-label="${esc(r.exit_label)}">▣</span>`;
  }

  // What a paper was predicted to need. Where no point estimate is on record the
  // band IS the value; putting a middot in the big-number slot and the real
  // content in the caption printed a punctuation mark as the headline.
  // the recorded band reads as a range, so the hyphen out of the data becomes an
  // en dash on the way to the page
  const predBand = (p) => (p && p.band ? String(p.band).replace(/\s*-\s*/, "–") : null);
  function predTile(p, tile) {
    const est = p.predicted_h100, band = predBand(p);
    if (est) return tile("PREDICTED NEED", R.fmtHM(est),
      band ? `band ${esc(band)} H100·h` : "H100·h", "");
    if (band) return tile("PREDICTED NEED", esc(band), "H100·h band", "");
    return tile("PREDICTED NEED", "not recorded", "no estimate for this paper", "quiet");
  }
  // the same value in a table cell. The unit is stated once in the column head,
  // so a band cell prints the range alone: repeating "H100·h" on the band rows
  // and not on the duration rows made one column read as two.
  function predCell(p) {
    const est = p.predicted_h100, band = predBand(p);
    if (est) return `<td class="num tnum c-pred" data-l="predicted">${R.fmtHM(est)}</td>`;
    if (band) return `<td class="num tnum c-pred" data-l="predicted"` +
      ` title="no point estimate on record; the paper's compute band is ${esc(band)} H100·h">` +
      `<span class="c-band">${esc(band)}</span></td>`;
    return `<td class="num tnum c-pred quiet" data-l="predicted">not recorded</td>`;
  }

  const scoreChip = (s) => (s == null ? "·"
    : `<span class="an-score ${s >= 8 ? "yes" : s >= 6 ? "over" : s <= 0 ? "no" : "slate"}">${esc(s)}<span>/10</span></span>`);

  const PapersView = {
    sortKey: "mean", sortDir: -1, q: "", _refocus: false,

    root() { return document.querySelector("#papers-root"); },

    // one row's worth of derived numbers, so sorting and drawing read the same table
    rows() {
      const S = window.State, models = window.Data.models;
      const byKey = {};
      window.Data.runs.forEach((r) => { byKey[r.arxiv_id + "|" + r.model] = r; });
      const q = this.q.trim().toLowerCase();
      return window.Data.papers
        .filter((p) => S.tier === "all" || p.tier === S.tier)
        .filter((p) => !q || `${p.claim || ""} ${p.arxiv_id} ${p.gist || ""}`.toLowerCase().includes(q))
        .map((p) => {
          const runs = models.map((m) => byKey[p.arxiv_id + "|" + m.key] || null);
          const scored = runs.filter((r) => r && r.audit_score != null);
          const mean = scored.length ? scored.reduce((a, r) => a + r.audit_score, 0) / scored.length : null;
          return { p, runs, mean, n: scored.length };
        });
    },

    cols() {
      // the compute unit is stated once, in the head, so the cells under it can
      // be bare numbers whether the record holds a duration or a band
      return [{ k: "claim", l: "claim" }, { k: "arxiv_id", l: "paper" }, { k: "tier", l: "tier" },
        { k: "predicted", l: "predicted", u: "H100·h", num: true }]
        .concat(window.Data.models.map((m) => ({ k: "m:" + m.key, l: m.name, num: true, model: m.key })))
        .concat([{ k: "mean", l: "mean", num: true }]);
    },

    sorted(rows) {
      const k = this.sortKey, dir = this.sortDir;
      const models = window.Data.models;
      const num = (row) => {
        if (k === "mean") return row.mean == null ? -1 : row.mean;
        if (k === "predicted") return row.p.predicted_h100 == null ? -1 : row.p.predicted_h100;
        if (k === "tier") return window.Data.tierRank(row.p.tier);
        if (k.startsWith("m:")) {
          const i = models.findIndex((m) => m.key === k.slice(2));
          const r = row.runs[i];
          return r && r.audit_score != null ? r.audit_score : -1;
        }
        return null;
      };
      return rows.sort((a, b) => {
        const x = num(a);
        const c = x == null
          ? String(a.p[k] || "").localeCompare(String(b.p[k] || "")) * dir
          : (x - num(b)) * dir;
        return c || String(a.p.arxiv_id).localeCompare(String(b.p.arxiv_id));
      });
    },

    headHtml() {
      return this.cols().map((c) => {
        const on = this.sortKey === c.k;
        const dir = this.sortDir < 0 ? "descending" : "ascending";
        return `<th class="${c.num ? "num" : ""} ${c.model ? "pm-head" : ""} ${on ? "sorted" : ""}` +
          `${c.model && window.State.model !== "all" && window.State.model !== c.model ? " dim" : ""}"` +
          ` data-k="${esc(c.k)}" tabindex="0" role="button" aria-sort="${on ? dir : "none"}"` +
          ` title="sort by ${esc(c.l)}">${esc(c.l)}${c.u ? `<span class="th-u"> · ${esc(c.u)}</span>` : ""}` +
          `<span class="th-dir">${on ? (this.sortDir < 0 ? "▼" : "▲") : ""}</span></th>`;
      }).join("");
    },

    renderList() {
      const host = this.root();
      if (!host) return;
      host.classList.add("wide-list");   // the grid is one column per agent, wider than prose
      const S = window.State, models = window.Data.models;
      const rows = this.sorted(this.rows());
      const nCols = 5 + models.length;
      const body = rows.map(({ p, runs, mean }) => {
        const hint = p.gist ? `${p.claim || p.arxiv_id}\n\n${p.gist}` : (p.claim || p.arxiv_id);
        return `<tr data-arx="${esc(p.arxiv_id)}" tabindex="0">
          <td class="pl-claim-cell c-claim"><span class="pcell-claim" title="${esc(hint)}">${esc(p.claim || p.arxiv_id)}</span></td>
          <td class="s-rid c-paper">${esc(p.arxiv_id)}</td>
          <td class="c-tier"><span class="badge ${TIER_CLS[p.tier] || "slate"}">${esc(window.Data.tierName(p.tier))}</span></td>
          ${predCell(p)}
          ${runs.map((r, i) => cellHtml(r, S.model !== "all" && S.model !== models[i].key, models[i])).join("")}
          <td class="num tnum pm-mean c-mean" data-l="mean" title="mean of the graded cells in this row">${mean == null ? "·" : mean.toFixed(2)}</td></tr>`;
      }).join("");
      host.innerHTML = `
        <div class="ov-head"><div><h1>Papers</h1>
          <div class="ov-sub">Every target paper by the claim the agent was asked to reproduce, one cell per agent carrying its audit score.</div></div></div>
        <div class="ps-head"><span class="plate">papers</span>
          <span class="ps-count">${this.q || S.tier !== "all" ? `${rows.length} of ${window.Data.papers.length}` : `${rows.length} papers`}</span>
          <span class="pm-legend">${["reproduced", "miss", "fault"].map((f) =>
            `<span class="lg-item ${f}"><span class="pm-chip ${f} sw-chip">·</span>${esc(V().meta(f).word)}</span>`).join("")}</span>
          <div class="ps-filters"><input id="pp-q" class="s-search" type="search" placeholder="search papers…" aria-label="search papers" value="${esc(this.q)}" /></div></div>
        <div class="tscroll"><table class="stats-table papers-table"><thead><tr>${this.headHtml()}</tr></thead>
          <tbody>${body || `<tr><td colspan="${nCols}" class="empty small">No papers match this search.` +
            `<span>Clear the search, or set the tier back to all tiers.</span></td></tr>`}</tbody></table></div>`;

      const s = host.querySelector("#pp-q");
      if (s) {
        if (this._refocus) { s.focus(); s.setSelectionRange(s.value.length, s.value.length); this._refocus = false; }
        s.oninput = () => { this.q = s.value; this._refocus = true; this.renderList(); };
      }
      host.querySelectorAll("thead th[data-k]").forEach((th) => {
        const sort = () => {
          const k = th.dataset.k;
          if (this.sortKey === k) this.sortDir *= -1;
          else { this.sortKey = k; this.sortDir = this.cols().find((c) => c.k === k).num ? -1 : 1; }
          this.renderList();
        };
        th.onclick = sort;
        th.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } };
      });
      host.querySelectorAll("tbody tr[data-arx]").forEach((tr) => {
        const open = () => window.go("#/paper/" + encodeURIComponent(tr.dataset.arx));
        tr.onclick = open;
        tr.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); open(); } };
      });
      host.querySelectorAll(".pm-chip[data-run]").forEach((b) => {
        b.onclick = (e) => { e.stopPropagation(); window.go("#/run/" + encodeURIComponent(b.dataset.run)); };
      });
    },

    // the paper's own instrument strip: what it was predicted to cost, how many
    // agents attempted it, how they scored, whether any of them landed it
    tilesHtml(p, runs) {
      const scored = runs.filter((r) => r.audit_score != null);
      const mean = scored.length ? scored.reduce((a, r) => a + r.audit_score, 0) / scored.length : null;
      const rep = runs.filter((r) => V().ofRun(r) === "reproduced").length;
      const tile = (l, v, sub, cls) =>
        `<div class="itile ${cls || ""}"><div class="it-l">${l}</div><div class="it-v tnum">${v}</div>` +
        `${sub ? `<div class="it-sub">${sub}</div>` : ""}</div>`;
      return `<div class="itiles">` +
        predTile(p, tile) +
        tile("AGENT RUNS", String(runs.length), `${scored.length} graded`, "") +
        tile("MEAN SCORE", mean == null ? "·" : mean.toFixed(2), "out of 10", "big") +
        tile("REPRODUCED", `${rep}/${runs.length}`, "graded 8 and above", rep ? "yes" : "") +
        `</div>`;
    },

    renderPaper(arx) {
      const host = this.root();
      if (!host) return;
      host.classList.remove("wide-list");   // one paper is a reading page, not a grid
      const p = window.Data.byPaper[arx];
      if (!p) { host.innerHTML = `<div class="empty">That paper is not in this collection.</div>`; return; }
      const tier = window.Data.tier(p.tier);
      const runs = window.Data.runsForPaper(arx)
        .sort((a, b) => (b.audit_score == null ? -1 : b.audit_score) - (a.audit_score == null ? -1 : a.audit_score));
      const fam = V().ofPaper(runs.some((r) => V().ofRun(r) === "reproduced"), runs.map((r) => V().ofRun(r)));
      const links = `${p.paper_url ? `<a href="${esc(p.paper_url)}" target="_blank" rel="noopener">paper ↗</a>` : ""}${p.code_url ? `<a href="${esc(p.code_url)}" target="_blank" rel="noopener">code ↗</a>` : ""}`;
      const rows = runs.map((r) => {
        const f = V().ofRun(r);
        const tok = (r.tokens && r.tokens.total) || null;
        const mode = window.Modes.name(r.mode);
        return `<tr data-run="${esc(r.id)}" tabindex="0">
          <td class="s-model c-agent"><span class="vg vd ${f}">${V().meta(f).glyph}</span>${esc(r.model_name)}</td>
          <td class="c-verdict">${V().inline(f, V().word(r))}</td>
          <td class="num c-score">${scoreChip(r.audit_score)}</td>
          <td class="mode-cell c-mode" title="${esc(mode)}">${window.Modes.chip(r.mode)}</td>
          <td class="fuel-cell c-fuel">${R.microFuelHtml(r)}</td>
          <td class="num tnum c-rounds" data-l="rounds">${exitMark(r)}${r.rounds == null ? "·" : esc(r.rounds)}</td>
          <td class="num tnum c-tokens" data-l="tokens">${R.fmtTok(tok)}</td></tr>`;
      }).join("");
      host.innerHTML = `<button class="crumb" id="pp-back">‹ all papers</button>
        <div class="spec-hero"><div class="spec-main">
          <div class="spec-tags">${V().stamp(fam)}${p.tier ? `<span class="badge ${TIER_CLS[p.tier] || "slate"}">${esc(tier.name)}</span>` : ""}
            ${p.kind ? `<span class="schip">${esc(p.kind)}</span>` : ""}
            <a class="spec-arx" href="https://arxiv.org/abs/${esc(p.arxiv_id)}" target="_blank" rel="noopener">${esc(p.arxiv_id)} ↗</a>
            <span class="spec-links">${links}</span></div>
          <h1 class="spec-claim">${esc(p.claim || p.arxiv_id)}</h1>
          ${p.gist ? `<p class="an-d-gist">${esc(p.gist)}</p>` : ""}
          ${tier.what ? `<div class="spec-band"><span class="plate">tier</span> ${esc(tier.name)} · ${esc(tier.what)}</div>` : ""}
        </div></div>
        ${this.tilesHtml(p, runs)}
        <div class="ps-head"><span class="plate">runs on this paper</span><span class="ps-count">${runs.length} of ${window.Data.models.length} agents</span></div>
        <div class="tscroll tshort"><table class="stats-table runs-table paper-runs"><thead><tr><th>agent</th><th>verdict</th><th class="num">score</th><th class="mode-head">failure mode</th><th>spent / budget</th><th class="num">rounds</th><th class="num">tokens</th></tr></thead>
          <tbody>${rows || `<tr><td colspan="7" class="empty small">No runs on this paper.</td></tr>`}</tbody></table></div>`;
      host.querySelector("#pp-back").onclick = () => window.go("#/papers");
      host.querySelectorAll("tbody tr[data-run]").forEach((tr) => {
        const open = () => window.go("#/run/" + encodeURIComponent(tr.dataset.run));
        tr.onclick = open;
        tr.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); open(); } };
      });
    },
  };

  window.PapersView = PapersView;
})();
