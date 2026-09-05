/* runcard.js: the small telemetry helpers shared by the runs table and the paper
   page: the agent short name and the micro fuel bar (spent against budget with an
   indigo tick at the paper's predicted need, readable H100·h numbers, no raw
   decimals). Attaches onto window.RENDER.

   Most runs spend a few percent of their budget, so the fill is drawn on a
   square-root scale and carries a floor: the low end of the column separates
   instead of collapsing onto one pixel, a nonzero spend is always a visible bar,
   and a run that spent nothing stays empty. The predicted tick is drawn only
   when the paper carries its own estimate and that estimate sits inside the
   budget; otherwise the tick would land on the end of the bar and mean nothing.

   The two durations are separate cells rather than one run of text, so live.css
   can pin each to a fixed width and keep the track, the slash and both numbers
   on the same x down a column of 275 rows. */
"use strict";

(function () {
  const R = window.RENDER, esc = R.esc;
  const clamp = (v) => Math.max(0, Math.min(100, v));
  // The track is drawn on a square-root scale. Most runs finish on a few percent
  // of their allowance, and linearly those rows are all the same invisible
  // sliver: 0.4% and 4% both round to under a pixel of an 84px rail. Rooted, 1%
  // is a tenth of the track and 4% a fifth, so the low end separates while the
  // full track still means the whole budget. The tick moves with the fill, so
  // spend against predicted still reads off one scale. Numbers stay literal.
  const scale = (v) => Math.sqrt(clamp(v) / 100) * 100;

  // micro fuel bar = spent/budget + indigo predicted tick; omitted with no budget data
  function microFuelHtml(run) {
    const { total, spent } = R.fuelNums(run);
    if (total == null || spent == null) return "";
    const pred = R.num(run.predicted_h100);
    const usable = pred != null && pred > 0 && total && pred < total;
    const over = (usable && spent > pred) || (total && spent > total);
    const pct = total ? clamp((spent / total) * 100) : 0;
    const predPct = usable ? clamp((pred / total) * 100) : null;
    const fillCls = (spent > 0 ? (over ? "over" : "done") : "zero");
    const title = `${R.fmtHM(spent)} of ${R.fmtHM(total)} H100·h · ${pct < 1 && spent > 0 ? "<1" : Math.round(pct)}% of budget` +
      (usable ? ` · predicted ${R.fmtHM(pred)}` : "");
    return `<span class="mfuel"><span class="mfuel-bar" title="${esc(title)}">` +
      `<i class="mfuel-fill ${fillCls}" style="width:${scale(pct).toFixed(1)}%"></i>` +
      (predPct != null ? `<i class="mfuel-pred" style="left:${scale(predPct).toFixed(1)}%"></i>` : "") +
      `</span><span class="mfuel-v tnum"><span class="mfuel-a">${R.fmtHM(spent)}</span>` +
      `<i>/</i><span class="mfuel-b">${R.fmtHM(total)}</span></span></span>`;
  }

  Object.assign(window.RENDER, { microFuelHtml });
})();
