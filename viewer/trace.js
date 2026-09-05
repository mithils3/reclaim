/* trace.js: the signature visual. One dependency-free inline-SVG primitive: a run's
   cumulative compute (H100·h) plotted against its round arc, raced against the
   indigo dashed "predicted" ceiling. The cyan curve recolours to amber the instant
   cumulative spend crosses that ceiling, and is capped by a verdict glyph. The
   same draw fn renders a card thumbnail and the full vertical transcript rail
   (which doubles as the minimap). Markers can be ranked by interest so a long
   routine stretch recedes to hairline ticks and the faults and spikes stand out;
   an optional invisible hit circle keeps every round clickable at that size.
   Also the dual-track took/predicted ledger bar. Colours go through style="" (var()
   is unreliable in SVG presentation attributes). */
"use strict";

(function () {
  const fill = (v) => `style="fill:var(${v})"`;
  const stroke = (v, extra) => `style="stroke:var(${v})${extra || ""}"`;

  function cumulative(rounds) {
    let cum = 0, maxDelta = 0;
    const pts = [];
    (rounds || []).forEach((r, i) => {
      let d = 0, fault = false;
      for (const c of (r.calls || [])) {
        if (typeof c.cost_h100 === "number") d += c.cost_h100;
        if (c.ok === false || c.error) fault = true;
      }
      cum = Math.round((cum + d) * 1e6) / 1e6;
      if (d > maxDelta) maxDelta = d;
      pts.push({ i, y: cum, delta: d, fault, kind: r.kind });
    });
    return { pts, total: cum, maxDelta };
  }

  // the value axis tops out at a round number a little above the curve and the
  // ceiling, so the panel can print the axis top and have it read as a number
  function niceMax(v) {
    if (!(v > 0)) return 1;
    const e = Math.pow(10, Math.floor(Math.log10(v))), m = v / e;
    const n = m <= 1 ? 1 : m <= 1.25 ? 1.25 : m <= 1.5 ? 1.5 : m <= 2 ? 2 : m <= 2.5 ? 2.5
      : m <= 3 ? 3 : m <= 4 ? 4 : m <= 5 ? 5 : m <= 6 ? 6 : m <= 8 ? 8 : 10;
    return Math.round(n * e * 1e6) / 1e6;
  }
  function yMaxOf(total, pred) {
    const raw = Math.max(total || 0, pred || 0, 1e-9);
    return Math.max(niceMax(raw * 1.06), raw * 1.04);
  }
  // Pinning the axis to the ceiling is only readable while the curve reaches a
  // fair share of it. One run in nine spends under a hundredth of its ceiling,
  // and on those the curve flattens onto the baseline under an empty plot with
  // the dashed ceiling stranded at the top. Below a fifth, the axis is scaled to
  // the spend instead and the caller reports the ceiling as off this scale.
  const OFF_SCALE = 0.2;
  function scaleOf(total, pred) {
    const t = total > 0 ? total : 0;
    const p = (typeof pred === "number" && pred > 0) ? pred : null;
    const off = !!(p && t > 0 && t < p * OFF_SCALE);
    return { yMax: off ? yMaxOf(t * 1.25, 0) : yMaxOf(t, p), off };
  }

  function splitAt(pts, pred) {
    if (pred == null || !pts.length) return { before: pts, after: [] };
    const before = [], after = [];
    for (let k = 0; k < pts.length; k++) {
      const p = pts[k];
      (p.y <= pred ? before : after).push(p);
      const q = pts[k + 1];
      if (q && (p.y <= pred) !== (q.y <= pred)) {
        const f = (pred - p.y) / (q.y - p.y);
        const cx = { i: p.i + (q.i - p.i) * f, y: pred };
        before.push(cx); after.unshift(cx);
      }
    }
    return { before, after };
  }

  function capSvg(fam, x, y, r, live) {
    const cv = { reproduced: "--yes", miss: "--over", fault: "--no", idle: "--slate", done: "--slate", running: "--accent" }[fam] || "--slate";
    const ember = live ? `<circle cx="${x}" cy="${y}" r="${r + 1.5}" class="trace-ember" ${fill(cv)}/>` : "";
    if (fam === "fault")
      return `${ember}<g ${stroke(cv)} stroke-width="${(r * .72).toFixed(1)}" stroke-linecap="round"><line x1="${x - r}" y1="${y - r}" x2="${x + r}" y2="${y + r}"/><line x1="${x - r}" y1="${y + r}" x2="${x + r}" y2="${y - r}"/></g>`;
    if (fam === "idle" || fam === "done")
      return `${ember}<circle cx="${x}" cy="${y}" r="${r}" style="fill:var(--panel);stroke:var(${cv})" stroke-width="1.3"/>`;
    if (fam === "miss")
      return `${ember}<circle cx="${x}" cy="${y}" r="${r}" ${fill(cv)}/><path d="M${x} ${y - r} A ${r} ${r} 0 0 1 ${x} ${y + r} Z" style="fill:var(--panel);opacity:.55"/>`;
    return `${ember}<circle cx="${x}" cy="${y}" r="${r}" ${fill(cv)}/>`;
  }

  function draw(rounds, opts) {
    opts = opts || {};
    const horiz = opts.orientation !== "v";
    const vbW = opts.vbW || (horiz ? 240 : 60);
    const vbH = opts.vbH || (horiz ? 80 : 320);
    const pad = opts.pad || (horiz ? 6 : 7);
    const { pts, total, maxDelta } = cumulative(rounds);
    // a transcript with nothing metered has no race to draw, so the ceiling is
    // dropped rather than plotted against a curve that never leaves zero
    const metered = total > 0;
    const ceiling = (metered && typeof opts.predicted === "number" && opts.predicted > 0) ? opts.predicted : null;
    // below a fifth of the ceiling the axis follows the spend and the ceiling
    // leaves the plot; the panel says so in its key
    const sc = scaleOf(total, ceiling);
    const pred = sc.off ? null : ceiling;
    const n = pts.length;
    // nothing to plot: say so on the chart rather than shipping a blank box
    if (!n)
      return `<svg class="trace empty-trace ${horiz ? "trace-h" : "trace-v"}" viewBox="0 0 ${vbW} ${vbH}" ` +
        `preserveAspectRatio="xMidYMid meet" role="img" aria-label="no rounds recorded">` +
        `<text class="empty-t" x="${(vbW / 2).toFixed(1)}" y="${(vbH / 2).toFixed(1)}" text-anchor="middle" ` +
        `dominant-baseline="central">no rounds recorded</text></svg>`;
    // a transcript with no metered compute (the auditor's own, say) has no value
    // axis to speak of, so the curve rides the middle and reads as a round index
    const flat = !metered;
    const yMax = sc.yMax;
    const vf = (y) => (flat ? 0.5 : y / yMax);
    const aMin = pad, aW = vbW - 2 * pad, aH = vbH - 2 * pad;
    const t = (i) => (n === 1 ? 0.5 : i / (n - 1));
    const map = horiz
      ? (i, y) => [aMin + t(i) * aW, aMin + aH - vf(y) * aH]
      : (i, y) => [aMin + vf(y) * aW, aMin + t(i) * aH];
    const poly = (arr) => arr.map((p) => map(p.i, p.y).map((v) => Math.round(v * 10) / 10).join(",")).join(" ");
    const sw = opts.sw || (horiz ? 2 : 2.2);

    let predSvg = "";
    if (pred != null) {
      const [px, py] = map(0, pred);
      // opts.band washes the allowance (everything under the ceiling) so the gap
      // between a thrifty curve and the paper's estimate reads as headroom the
      // run did not use rather than as an empty plot
      const band = opts.band
        ? (horiz
          ? `<rect x="${aMin}" y="${py.toFixed(1)}" width="${aW}" height="${(aMin + aH - py).toFixed(1)}" style="fill:var(--predicted-tint);opacity:.38"/>`
          : `<rect x="${aMin}" y="${aMin}" width="${(px - aMin).toFixed(1)}" height="${aH}" style="fill:var(--predicted-tint);opacity:.38"/>`)
        : "";
      predSvg = band + (horiz
        ? `<line x1="${aMin}" y1="${py.toFixed(1)}" x2="${vbW - pad}" y2="${py.toFixed(1)}" ${stroke("--predicted", ";opacity:.85")} stroke-width="1" stroke-dasharray="3 3"/>`
        : `<line x1="${px.toFixed(1)}" y1="${aMin}" x2="${px.toFixed(1)}" y2="${vbH - pad}" ${stroke("--predicted", ";opacity:.85")} stroke-width="1" stroke-dasharray="3 3"/>`);
    }
    let fillSvg = "";
    if (opts.fill) {
      const base = horiz ? `${aMin + aW},${aMin + aH} ${aMin},${aMin + aH}` : `${aMin},${aMin + aH} ${aMin},${aMin}`;
      fillSvg = `<polygon points="${poly(pts)} ${base}" ${fill("--accent-tint")}/>`;
    }
    const { before, after } = splitAt(pts, pred);
    const cyan = before.length > 1 ? `<polyline points="${poly(before)}" fill="none" ${stroke("--accent")} stroke-width="${sw}" stroke-linejoin="round" stroke-linecap="round"/>` : "";
    const amber = after.length > 1 ? `<polyline points="${poly(after)}" fill="none" ${stroke("--over")} stroke-width="${sw}" stroke-linejoin="round" stroke-linecap="round"/>` : "";
    // markers. opts.interest ranks a round by how much it is worth looking at, so a
    // long routine stretch recedes to hairline ticks instead of fusing into one
    // solid bar and burying the faults and the spikes that carry the story.
    const spike = (typeof opts.spike === "number" && opts.spike > 0) ? opts.spike : maxDelta;
    // opts.density is how many px each round actually gets along the axis. Marker
    // radii scale with it and are capped below half that gap, so a 94-round rail
    // stays a row of separable dots and a 16-round rail gets dots worth clicking.
    const dens = (typeof opts.density === "number" && opts.density > 0) ? opts.density : null;
    const rScale = dens ? Math.max(.62, Math.min(1.45, dens / 8)) : 1;
    const rCap = dens ? Math.max(1, dens * .46) : Infinity;
    const rOf = (r) => Math.round(Math.min(r * rScale, rCap) * 100) / 100;
    // a transcript with no metered compute at all (the auditor's own) has no
    // interest ranking to make, so every round takes the same working marker
    // rather than the hairline tick that means "this round cost nothing"
    const unmetered = !(total > 0);
    function markOf(p) {
      const overHere = p.y > (pred || Infinity);
      if (!opts.interest) return { r: opts.markerR || 2, cv: p.fault ? "--no" : (overHere ? "--over" : "--accent"), cls: "" };
      if (p.fault) return { r: rOf(3.4), cv: "--no", cls: " mk-fault" };
      if (p.kind === "final") return { r: rOf(3.2), cv: overHere ? "--over" : "--accent", cls: " mk-final" };
      if (spike > 0 && p.delta >= spike * 0.5) return { r: rOf(2.9), cv: overHere ? "--over" : "--accent", cls: " mk-spike" };
      if (p.delta > 0 || unmetered) return { r: rOf(1.9), cv: overHere ? "--over" : "--accent", cls: " mk-work" };
      return { r: rOf(1.15), cv: "--slate", cls: " mk-idle" };
    }
    let marks = "";
    if (opts.markers) {
      // a per-round hit target, painted last so a hairline tick is still clickable
      const hit = opts.hit ? Math.max(3.4, Math.min(9, ((horiz ? aW : aH) / Math.max(n - 1, 1)) / 1.8)) : 0;
      // the tooltip names the recorded round, not the position in the array
      const idOf = typeof opts.idOf === "function" ? opts.idOf : (p) => p.i;
      const dots = [], hits = [];
      pts.forEach((p) => {
        const [mx, my] = map(p.i, p.y);
        const cx = Math.round(mx * 10) / 10, cy = Math.round(my * 10) / 10;
        const m = markOf(p);
        const tip = `round ${idOf(p)}${p.fault ? " · failed call" : ""}` +
          `${p.delta > 0 ? ` · +${Math.round(p.delta * 1e4) / 1e4} H100·h` : ""}`;
        dots.push(`<circle class="trace-mark${m.cls}" data-round="${p.i}" cx="${cx}" cy="${cy}" r="${m.r}" ${fill(m.cv)}><title>${tip.replace(/</g, "&lt;")}</title></circle>`);
        if (hit) hits.push(`<circle class="trace-hit" data-round="${p.i}" cx="${cx}" cy="${cy}" r="${hit.toFixed(1)}"><title>${tip.replace(/</g, "&lt;")}</title></circle>`);
      });
      marks = dots.join("") + hits.join("");
    }
    const axis = opts.axis
      ? (horiz
        ? `<line class="trace-ax" x1="${aMin}" y1="${aMin + aH}" x2="${vbW - pad}" y2="${aMin + aH}"/>`
        : `<line class="trace-ax" x1="${aMin}" y1="${aMin}" x2="${aMin}" y2="${vbH - pad}"/>`)
      : "";
    const last = pts[n - 1], [lx, ly] = map(last.i, last.y);
    const cap = capSvg(opts.family || "done", Math.round(lx * 10) / 10, Math.round(ly * 10) / 10, opts.capR || (horiz ? 3 : 3.4), opts.live);
    const lab = opts.label ? ` role="img" aria-label="${String(opts.label).replace(/"/g, "&quot;")}"` : ` aria-hidden="true"`;
    return `<svg class="trace ${horiz ? "trace-h" : "trace-v"}" viewBox="0 0 ${vbW} ${vbH}" preserveAspectRatio="xMidYMid meet"${lab}>${axis}${predSvg}${fillSvg}${cyan}${amber}${marks}${cap}</svg>`;
  }

  function ledger(took, predicted, opts) {
    opts = opts || {};
    if (took == null && predicted == null) return "";
    const tk = took || 0, pr = predicted || 0;
    const max = Math.max(tk, pr, 1e-9);
    const predPct = (pr / max) * 100, tookPct = (tk / max) * 100;
    const overPct = Math.max(0, tookPct - predPct), basePct = Math.min(tookPct, predPct);
    const tick = pr > 0 ? `<i class="lg-tick" style="left:${predPct.toFixed(1)}%"></i>` : "";
    let label = "";
    if (opts.pct && pr > 0 && took != null) {
      const r = Math.round((tk / pr) * 100);
      // a real spend never prints as 0%: the bar would read as nothing spent
      const s = (r === 0 && tk > 0) ? "<1%" : `${r}%`;
      label = `<span class="lg-pct tnum ${tk > pr ? "over" : "under"}">${s}</span>`;
    }
    return `<span class="ledger ${opts.size || ""}"><span class="lg-track">` +
      (pr > 0 ? `<i class="lg-pred" style="width:${predPct.toFixed(1)}%"></i>` : "") +
      `<i class="lg-took" style="width:${basePct.toFixed(1)}%"></i>` +
      (overPct > 0 ? `<i class="lg-over" style="left:${predPct.toFixed(1)}%;width:${overPct.toFixed(1)}%"></i>` : "") +
      `${tick}</span>${label}</span>`;
  }

  window.Trace = { cumulative, draw, ledger, yMaxOf, scaleOf };
})();
