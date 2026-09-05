/* charts.js: dependency-free themed visuals. ArcGauge for the headline rate, a
   generic horizontal bar grammar (card + rows + legend) reused by the
   failure-mode stack and the budget row, and a vertical column histogram
   for the 0 to 10 score distribution. All colour goes through the design tokens, so
   dark mode works for free. The burn trace and ledger live in trace.js. */
"use strict";

(function () {
  const esc = window.RENDER.esc;

  // ---- ArcGauge: 270° gauge, progress over a track, numerals centered --------
  // The box is cropped to the arc it draws, so the ring uses the whole width it
  // is given and the caption below it starts where the stroke ends. The old box
  // carried 16 units of blank under the opening and a ring narrow enough that a
  // six-character numerator crossed the stroke on both sides.
  const GB = { w: 118, h: 107, cx: 59, cy: 59, r: 52, sw: 10, a0: 135, sweep: 270 };
  function arcGauge(numer, denom, opts) {
    opts = opts || {};
    const { cx, cy, r, sw, a0: A0, sweep: SWEEP } = GB;
    const pct = denom ? numer / denom : 0;
    const polar = (deg) => { const a = deg * Math.PI / 180; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; };
    const arc = (a0, a1) => { const [x0, y0] = polar(a0), [x1, y1] = polar(a1); const large = (a1 - a0) > 180 ? 1 : 0;
      return `M${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`; };
    const prog = A0 + SWEEP * Math.max(0, Math.min(1, pct));
    const numTxt = String(opts.num ?? `${numer}/${denom}`);
    const pctTxt = opts.pct ?? `${denom ? Math.round(pct * 100) : 0}%`;
    const label = opts.label ?? `${numer} of ${denom}`;
    // the numerals sit inside the ring, so they shrink as the string grows: the
    // clear width is the inner diameter less an 8-unit margin on each side, and
    // Fraunces 700 measures about .62em to the character at these sizes
    const clear = 2 * (r - sw / 2) - 16;
    const nSize = Math.min(24, Math.floor(clear / (0.62 * Math.max(1, numTxt.length))));
    return `<svg class="arc-gauge" viewBox="0 0 ${GB.w} ${GB.h}" role="img" aria-label="${esc(label)}">
      <path d="${arc(A0, A0 + SWEEP)}" class="ag-track" fill="none" stroke-width="${sw}" stroke-linecap="round"/>
      ${pct > 0 ? `<path d="${arc(A0, prog)}" class="ag-prog" fill="none" stroke-width="${sw}" stroke-linecap="round"/>` : ""}
      <text x="${cx}" y="${cy}" text-anchor="middle" class="ag-num" style="font-size:${nSize}px">${esc(numTxt)}</text>
      <text x="${cx}" y="${cy + 22}" text-anchor="middle" class="ag-pct">${esc(pctTxt)}</text>
    </svg>`;
  }

  // ---- horizontal bar grammar ----------------------------------------------
  // cls marks a card apart from its neighbours, e.g. one that must keep its
  // natural height instead of stretching to the row
  const card = (title, sub, legend, inner, cls) =>
    `<div class="chart-card ${cls || ""}"><div class="chart-h"><span class="chart-t">${esc(title)}</span>${sub ? `<span class="chart-s">${esc(sub)}</span>` : ""}</div>${legend || ""}<div class="bars">${inner}</div></div>`;
  const empty = (title) => card(title, "", "", `<div class="chart-empty">no data</div>`);

  // segs: [{cls | colour, pct, title}]. cls picks a token class, colour a css var.
  // rowCls marks a row apart from the others, e.g. an all-agents total.
  function barRow(label, segs, valTxt, tip, rowCls) {
    const segHtml = segs.map((s) => {
      const style = s.colour ? `background:var(${s.colour});` : "";
      return `<span class="seg ${s.cls || ""}" style="${style}width:${s.pct.toFixed(2)}%" title="${esc(s.title || "")}"></span>`;
    }).join("");
    return `<div class="bar-row ${rowCls || ""}"><div class="bar-label" title="${esc(tip || label)}">${esc(label)}</div>` +
      `<div class="bar-track">${segHtml}</div><div class="bar-val tnum">${esc(valTxt)}</div></div>`;
  }
  // items: [{colour, label, title, n}]. The title carries the long form, so a
  // legend entry can stay one short line and still be readable on hover. An
  // item with n prints its count in the mono numeral column at the right of the
  // entry, which is what lets a ten-entry legend be read as a table instead of
  // as a paragraph of dots. opts.cls adds a layout variant, e.g. "lg-grid".
  function legend(items, opts) {
    opts = opts || {};
    return `<div class="chart-legend${opts.cls ? " " + opts.cls : ""}">${items.map((i) =>
      `<span class="lg ${i.cls || ""}"${i.title ? ` title="${esc(i.title)}"` : ""}>` +
      `<i class="sw" style="background:var(${i.colour})"></i>` +
      `<span class="lg-t">${esc(i.label)}</span>` +
      (i.n == null ? "" : `<b class="lg-n tnum">${esc(String(i.n))}</b>`) +
      `</span>`).join("")}</div>`;
  }

  // ---- vertical column histogram (0 to 10 scores) ---------------------------
  // opts.axis names the horizontal axis under the keys; a column chart with no
  // named axis leaves the reader to guess what the numerals along the bottom are
  function histogram(counts, opts) {
    opts = opts || {};
    const keys = opts.keys || Object.keys(counts);
    const max = Math.max(1, ...keys.map((k) => counts[k] || 0));
    const cols = keys.map((k) => {
      const v = counts[k] || 0;
      const h = (v / max) * 100;
      const cv = opts.colour ? opts.colour(k) : "--accent";
      return `<div class="hg-col" title="${esc(k)}: ${v} run${v === 1 ? "" : "s"}">` +
        `<div class="hg-n tnum">${v || ""}</div>` +
        `<div class="hg-bar"><i style="height:${h.toFixed(1)}%;background:var(${cv})"></i></div>` +
        `<div class="hg-k tnum">${esc(k)}</div></div>`;
    }).join("");
    return `<div class="hgram">${cols}</div>` +
      (opts.axis ? `<div class="hg-axis">${esc(opts.axis)}</div>` : "");
  }

  window.Charts = { arcGauge, card, empty, barRow, legend, histogram };
})();
