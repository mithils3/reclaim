/* strip.js: the transcript's vertical burn-trace rail behaviour (drawing is
   trace.js). The rail IS the scrollbar and minimap: it is capped with the round
   count so it reads as one before it is hovered, each round is a marker on the
   cumulative-compute curve ranked by interest so faults and spikes stand out of a
   long routine stretch, clicking one scrolls that round into view (opening its
   group if it sits in one), and an IntersectionObserver slides a "you-are-here"
   tick as you scroll. A jump-to-verdict button drops you on the FINAL round. */
"use strict";

(function () {
  const Strip = {
    rootEl: null, run: null, rounds: null, rail: null, _io: null, _resize: null, _marks: null,

    cleanup() {
      if (this._io) { this._io.disconnect(); this._io = null; }
      if (this._resize) { window.removeEventListener("resize", this._resize); this._resize = null; }
    },
    mount(rootEl, run, rounds) {
      this.cleanup();
      this.rootEl = rootEl; this.run = run; this.rounds = rounds || [];
      this.rail = rootEl.querySelector(".strip-rail");
      if (!this.rail || !this.rounds.length) return;
      this._resize = () => this.draw();
      window.addEventListener("resize", this._resize);
      this.draw(); this.observe();
    },

    draw() {
      const rail = this.rail; if (!rail || !window.Trace) return;
      const n = this.rounds.length;
      // the rail says what it is before it is hovered: a labelled count on top, the
      // curve in the middle, the jump-to-verdict control on the foot
      rail.innerHTML = `<div class="rail-inner">
        <div class="rail-cap"><b class="tnum">${n}</b><span>rounds</span></div>
        <div class="rail-plot" title="cumulative compute per round · click a round to jump to it"><i class="rail-here" hidden></i></div>
        <div class="rail-foot"><button class="rail-jump" title="jump to the final verdict" aria-label="jump to the final verdict">⤓</button></div>
      </div>`;
      const plot = rail.querySelector(".rail-plot");
      const w = plot.clientWidth || 64, h = plot.clientHeight || 420;
      const predicted = window.RENDER.predictedOf(this.run);
      const family = window.Verdict ? window.Verdict.ofRun(this.run) : "done";
      // markers scale with how much room each round actually gets, so a 94-round
      // transcript stays a row of separable dots instead of one fused bar
      const gap = (h - 16) / Math.max(n - 1, 1);
      const svg = window.Trace.draw(this.rounds, { orientation: "v", vbW: w, vbH: h, pad: 8,
        predicted, family, markers: true, interest: true, hit: true, axis: true,
        density: gap, spike: this.run && this.run.__spike, capR: 3.6,
        idOf: (p) => { const r = this.rounds[p.i]; return r && r.round_index != null ? r.round_index : p.i; },
        label: `burn trace over ${n} rounds` });
      plot.insertAdjacentHTML("afterbegin", svg);
      // markers carry their position in the rounds array; the round cards carry
      // the recorded round_index. Keep both maps so the two never drift apart.
      this._marks = {}; this._posOf = {};
      this.rounds.forEach((r, i) => { this._posOf[String(r.round_index)] = String(i); });
      plot.querySelectorAll(".trace-mark").forEach((mk) => { this._marks[mk.dataset.round] = mk; });
      plot.querySelectorAll("[data-round]").forEach((mk) => {
        const pos = mk.dataset.round;
        mk.addEventListener("click", () => this.scrollToPos(pos));
        mk.addEventListener("mouseenter", () => this.hover(pos, true));
        mk.addEventListener("mouseleave", () => this.hover(pos, false));
      });
      const jb = rail.querySelector(".rail-jump");
      if (jb) jb.addEventListener("click", () => this.scrollToVerdict());
    },

    hover(pos, on) {
      const mk = this._marks && this._marks[pos];
      if (mk) mk.classList.toggle("hover", !!on);
    },

    scrollToPos(pos) {
      const r = this.rounds[+pos];
      if (r) this.scrollToRound(r.round_index);
    },
    scrollToRound(idx) {
      const card = window.RENDER.revealRound(this.rootEl, idx);
      if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    },
    scrollToVerdict() {
      const f = this.rootEl.querySelector(".rcard.final") || this.rootEl.querySelector(".rounds .rcard:last-child");
      if (f) f.scrollIntoView({ behavior: "smooth", block: "center" });
    },

    // observes the top-level children of .rounds, so a collapsed group of routine
    // rounds still moves the you-are-here tick (it carries its first round index)
    observe() {
      if (this._io) this._io.disconnect();
      const root = this.rootEl.closest(".detail") || null;
      this._io = new IntersectionObserver((entries) => {
        let best = null;
        for (const e of entries) if (e.isIntersecting && (!best || e.intersectionRatio > best.intersectionRatio)) best = e;
        if (best) this.setHere(best.target.getAttribute("data-round"));
      }, { root, threshold: [0, .35, .7, 1], rootMargin: "-42% 0px -42% 0px" });
      this.rootEl.querySelectorAll(".rounds > .rcard, .rounds > .rgroup, .rgroup:not(.collapsed) .rcard")
        .forEach((c) => this._io.observe(c));
    },
    setHere(idx) {
      const rail = this.rail; if (!rail || idx == null) return;
      rail.querySelectorAll(".trace-mark.here").forEach((m) => m.classList.remove("here"));
      const pos = this._posOf && this._posOf[String(idx)];
      const mk = pos != null && this._marks ? this._marks[pos] : null;
      const here = rail.querySelector(".rail-here");
      const plot = rail.querySelector(".rail-plot");
      if (!mk || !here || !plot) return;
      mk.classList.add("here");
      // the viewBox is drawn at the plot's own pixel size, so cy is already in
      // plot pixels and the tick rides inside .rail-plot, not the whole rail
      here.hidden = false;
      here.style.top = parseFloat(mk.getAttribute("cy") || 0) + "px";
    },
  };
  window.Strip = Strip;
})();
