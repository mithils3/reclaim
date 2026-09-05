/* overview.js: the landing worksheet. A hero band that states the collection in
   one gauge, four numbers and the three tier definitions, then the agent x tier
   matrix (mean audit score, reproduced share, n) whose cells route into the
   filtered runs table, a failure-mode stack per agent in the fixed vocabulary
   order, the 0 to 10 score histogram and the budget row that puts reproduced
   runs next to the rest, tier by tier. The matrix always shows the full four by
   three plus an all-agents footer; everything else is recomputed from the run
   rows under the global Agent and Tier filter. Markup only: every class it
   writes is defined in styles.css, anon.css or overview.css. */
"use strict";

(function () {
  const R = window.RENDER, esc = R.esc, C = () => window.Charts;
  const TIER_CLS = { run: "tier-run", retrain: "tier-retrain", reimplement: "tier-reimplement" };
  const pct1 = (x) => (Math.round(x * 1000) / 10).toFixed(1) + "%";
  const pct0 = (x) => Math.round(x * 100) + "%";

  // the four score bands the auditor's verdict is derived from, in code
  function scoreColour(s) {
    const n = Number(s);
    return n >= 8 ? "--yes" : n >= 6 ? "--over" : n <= 0 ? "--no" : "--slate";
  }
  const SCORE_BANDS = [
    { colour: "--no", label: "disqualified (0)" },
    { colour: "--slate", label: "not reproduced (1 to 5)" },
    { colour: "--over", label: "partial (6 to 7)" },
    { colour: "--yes", label: "reproduced (8 to 10)" },
  ];

  const meanOf = (arr) => (arr.length ? arr.reduce((a, x) => a + x, 0) / arr.length : null);
  const scores = (rs) => rs.filter((r) => r.audit_score != null).map((r) => r.audit_score);
  const repro = (rs) => rs.filter((r) => r.audit_reproduced).length;
  // fraction of its own granted budget one run spent
  const spentFrac = (r) => (r.budget_h100 ? (r.spent_h100 || 0) / r.budget_h100 : null);

  const OverviewView = {
    root() { return document.querySelector("#overview-root"); },

    rows() {
      const D = window.Data, S = window.State;
      return D.runs.filter((r) =>
        (S.model === "all" || r.model === S.model) &&
        (S.tier === "all" || r.tier === S.tier));
    },

    /* the hero: the gauge holds the left column, the stat strip and the tier
       legend stack in the right one, so the tall card is matched by content
       rather than by empty stretch. Every number in the strip reads the current
       filter: papers and runs both narrow to the selection, so the strip cannot
       claim 100 papers beside 30 runs on one tier. */
    heroHtml(rows) {
      const D = window.Data, S = window.State;
      const all = D.runs.length, n = rows.length, filtered = n !== all;
      const papers = D.benchmark.papers != null ? D.benchmark.papers : D.papers.length;
      // under a filter the count is the papers the selected runs actually touch
      const attempted = new Set(rows.map((r) => r.arxiv_id)).size;
      const rep = repro(rows), sc = scores(rows), mean = meanOf(sc);
      const stat = (l, v, s, cls) =>
        `<div class="ov-stat ${cls || ""}"><div class="ovs-l">${esc(l)}</div>` +
        `<div class="ovs-v tnum">${esc(String(v))}</div><div class="ovs-s">${esc(s)}</div></div>`;
      const tiers = D.tiers.map((t) => {
        const off = S.tier !== "all" && S.tier !== t.key;
        return `<div class="ov-tierdef ${off ? "off" : ""}">` +
          `<span class="badge ${TIER_CLS[t.key] || "slate"}">${esc(t.name)}</span>` +
          `<span class="ovtd-what">${esc(t.what || "")}</span></div>`;
      }).join("");
      return `<section class="ov-hero">
        <div class="panel-card ov-gauge">
          ${C().arcGauge(rep, n || 1, {
            num: `${rep}/${n}`,
            pct: n ? pct1(rep / n) : "no runs",
            label: `${rep} of ${n} runs reproduced`,
          })}
          <div class="gauge-cap"><b>reproduced runs</b><span>audit score 8 and above</span></div>
        </div>
        <div class="panel-card ov-stats">
          ${stat("papers", filtered ? attempted : papers, filtered ? `of ${papers} attempted` : "target papers")}
          ${stat("runs", n, filtered ? `of ${all} published` : "one per paper and tier")}
          ${stat("agents", S.model === "all" ? D.models.length : 1,
            S.model === "all" ? "under test" : D.modelLabel(S.model))}
          ${stat("mean score", mean == null ? "·" : mean.toFixed(2), `over ${sc.length} graded runs`, "accent")}
        </div>
        <div class="panel-card ov-tierdefs">${tiers}</div>
      </section>`;
    },

    /* the matrix is the collection, not the selection: the filter dims the cells
       it excludes instead of removing them, so the four by three always reads.
       The last column and the last row are the margins, in the demoted grey, and
       every cell including a margin opens the runs behind it. */
    matrixHtml() {
      const D = window.Data, S = window.State;
      const cols = D.tiers.map((t) => ({ key: t.key, name: t.name, cls: TIER_CLS[t.key] || "slate" }))
        .concat([{ key: "all", name: "All tiers", cls: "slate", margin: true }]);
      const rows = D.models.map((m) => ({ key: m.key, name: m.name }))
        .concat([{ key: "all", name: "All agents", margin: true }]);
      const pick = (mk, tk) => D.runs.filter((r) =>
        (mk === "all" || r.model === mk) && (tk === "all" || r.tier === tk));
      const cell = (row, col) => {
        const rs = pick(row.key, col.key);
        const label = `${row.name} · ${col.name}`;
        if (!rs.length) return `<td class="mt-cell empty" data-label="${esc(col.name)}">·</td>`;
        const mean = meanOf(scores(rs)), rep = repro(rs);
        const share = (rep / rs.length) * 100;
        const off = (S.model !== "all" && !row.margin && S.model !== row.key) ||
                    (S.tier !== "all" && !col.margin && S.tier !== col.key);
        return `<td class="mt-cell live ${row.margin || col.margin ? "mt-total" : ""} ${off ? "off" : ""}"
            data-model="${esc(row.key)}" data-tier="${esc(col.key)}" data-label="${esc(col.name)}"
            tabindex="0" role="button"
            title="${esc(label)} · open the ${rs.length} run${rs.length === 1 ? "" : "s"} behind this cell">
          <span class="mt-mean tnum">${mean == null ? "·" : mean.toFixed(2)}</span>
          <span class="mt-frac tnum">${rep}/${rs.length} reproduced</span>
          <span class="mt-bar" role="img" aria-label="${share.toFixed(0)} percent reproduced">${
            share > 0 ? `<i style="width:${share.toFixed(0)}%"></i>` : ""}</span></td>`;
      };
      const head = `<thead><tr><th class="mt-corner"></th>${cols.map((c) =>
        `<th class="mt-tier" scope="col"><span class="badge ${c.cls}">${esc(c.name)}</span></th>`
      ).join("")}</tr></thead>`;
      const line = (row) => `<tr><th class="mt-model" scope="row">${esc(row.name)}</th>` +
        cols.map((c) => cell(row, c)).join("") + `</tr>`;
      const body = `<tbody>${rows.filter((r) => !r.margin).map(line).join("")}</tbody>`;
      const foot = `<tfoot>${rows.filter((r) => r.margin).map(line).join("")}</tfoot>`;
      return `<div class="panel-card ov-matrix">
        <div class="pc-head"><span class="plate">audit score by agent and tier</span><span class="lc-sub">mean out of 10 · the bar is the reproduced share · click a cell for its runs</span></div>
        <div class="tscroll"><table class="matrix mt-big">${head}${body}${foot}</table></div>
      </div>`;
    },

    modesHtml(rows) {
      const D = window.Data, S = window.State;
      const present = window.Modes.all.filter((m) => D.runs.some((r) => r.mode === m.key));
      if (!present.length || !rows.length) return C().empty("primary failure mode by agent");
      const total = {};
      rows.forEach((r) => { if (r.mode) total[r.mode] = (total[r.mode] || 0) + 1; });
      const agents = D.models.filter((m) => S.model === "all" || m.key === S.model);
      const segsOf = (rs) => {
        const counts = {};
        rs.forEach((r) => { const k = r.mode || "other"; counts[k] = (counts[k] || 0) + 1; });
        return present.filter((x) => counts[x.key]).map((x) => ({
          colour: x.colour, pct: (counts[x.key] / rs.length) * 100,
          title: `${x.name}: ${counts[x.key]} of ${rs.length} runs`,
        }));
      };
      const tiers = D.tiers.filter((t) => S.tier === "all" || t.key === S.tier);
      let bars = agents.map((m) => {
        const rs = rows.filter((r) => r.model === m.key);
        if (!rs.length) return "";
        return C().barRow(m.name, segsOf(rs), `${rs.length} runs`, `${m.name} · ${rs.length} runs`);
      }).join("");
      if (agents.length > 1) {
        bars += C().barRow("All agents", segsOf(rows), `${rows.length} runs`, `all agents · ${rows.length} runs`, "bar-total");
      } else if (tiers.length > 1) {
        // one agent leaves one bar, which says less than the same runs split by
        // tier: the stack per tier is the reading the selection was made for
        bars += tiers.map((t) => {
          const rs = rows.filter((r) => r.tier === t.key);
          if (!rs.length) return "";
          return C().barRow(t.name, segsOf(rs), `${rs.length} runs`, `${t.name} · ${rs.length} runs`, "bar-sub");
        }).join("");
      }
      // ten entries read as a two-column table with the counts in one mono
      // column, not as a paragraph of coloured dots wrapped over four lines
      const lg = C().legend(present.map((x) => ({
        colour: x.colour, label: x.name, n: total[x.key] || 0, title: x.definition,
        cls: total[x.key] ? "" : "lg-zero",   // the vocabulary stays whole; a mode
      })), { cls: "lg-grid" });               // absent from the selection recedes
      const sub = agents.length > 1 ? "share of each agent's runs"
        : tiers.length > 1 ? "share of this agent's runs, by tier" : "share of this agent's runs";
      // one agent on one tier is a single bar: the card keeps its own height
      // rather than stretching to the histogram beside it around a void
      const thin = agents.length === 1 && tiers.length === 1 ? "ov-modes-thin" : "";
      return C().card("primary failure mode by agent", sub, lg, bars, thin);
    },

    histogramHtml(rows) {
      const counts = {};
      for (let i = 0; i <= 10; i++) counts[i] = 0;
      let graded = 0;
      rows.forEach((r) => {
        if (r.audit_score != null) { counts[r.audit_score] = (counts[r.audit_score] || 0) + 1; graded++; }
      });
      const keys = Object.keys(counts).sort((a, b) => Number(a) - Number(b));
      return `<div class="chart-card ov-hist">
        <div class="chart-h"><span class="chart-t">score distribution</span><span class="chart-s">${graded} graded run${graded === 1 ? "" : "s"}</span></div>
        ${C().legend(SCORE_BANDS)}
        ${C().histogram(counts, { keys, colour: scoreColour, axis: "audit score, 0 to 10" })}</div>`;
    },

    /* budget: mean share of a run's own budget, reproduced runs against the
       rest. The three tiers stand side by side as three columns of the panel,
       which is what the panel is wide enough for; a single tier stacked down
       the middle left two thirds of every row empty. Like the matrix and the
       hero legend, the tier filter dims the columns it excludes rather than
       removing them, so the three-way comparison always reads. */
    computeHtml() {
      const D = window.Data, S = window.State;
      const base = D.runs.filter((r) => S.model === "all" || r.model === S.model);
      const groups = D.tiers.map((t) => {
        const rs = base.filter((r) => r.tier === t.key);
        const off = S.tier !== "all" && S.tier !== t.key;
        const line = (label, subset, colour) => {
          const fr = meanOf(subset.map(spentFrac).filter((x) => x != null));
          if (fr == null) return C().barRow(label, [], `· ${subset.length} runs`, `${t.name} · ${label} · no runs`);
          return C().barRow(label, [{ colour, pct: Math.max(0, Math.min(100, fr * 100)), title: `${pct0(fr)} of budget` }],
            `${pct0(fr)} · ${subset.length} runs`, `${t.name} · ${label} · mean ${pct0(fr)} of budget over ${subset.length} runs`);
        };
        const body = rs.length
          ? line("reproduced", rs.filter((r) => r.audit_reproduced), "--yes") +
            line("not reproduced", rs.filter((r) => !r.audit_reproduced), "--no")
          : `<div class="chart-empty">no runs</div>`;
        return `<div class="cmp-group ${off ? "off" : ""}">
          <div class="cmp-tier"><span class="badge ${TIER_CLS[t.key] || "slate"}">${esc(t.name)}</span><span class="cmp-what">${rs.length} runs</span></div>
          ${body}
        </div>`;
      }).join("");
      const sub = S.model === "all" ? "mean share of a run's own granted budget"
        : `mean share of a run's own granted budget · ${D.modelLabel(S.model)}`;
      return C().card("budget spent", sub, "", groups || `<div class="chart-empty">no data</div>`);
    },

    render() {
      const host = this.root();
      if (!host) return;
      const rows = this.rows();
      host.innerHTML = `
        <div class="ov-head">
          <div><h1>Reproduction run traces</h1>
            <div class="ov-sub">Agents attempting the central claim of a machine learning paper under a metered budget, graded by ${esc(window.Data.auditor.name)}.</div></div>
        </div>
        ${this.heroHtml(rows)}
        ${this.matrixHtml()}
        <section class="grid2 ov-grid">${this.modesHtml(rows)}${this.histogramHtml(rows)}</section>
        <section class="ov-compute">${this.computeHtml()}</section>`;
      host.querySelectorAll(".mt-cell.live").forEach((td) => {
        const open = () => window.go(window.runsHash({ model: td.dataset.model, tier: td.dataset.tier, verdict: "all", mode: "all", q: "" }));
        td.addEventListener("click", open);
        td.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      });
    },
  };

  window.OverviewView = OverviewView;
})();
