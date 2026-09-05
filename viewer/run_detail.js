/* run_detail.js: one run, end to end. The header card states the claim and the
   instruments, the audit card carries the grade the auditor gave and the flags it
   raised, the dissection card carries the post-hoc read of what the agent actually
   did, and the transcript below is the burn trace, the round rail and the round
   cards. Where the auditor's own transcript was captured, a toggle swaps it in
   through the same round renderer. Previous and next follow the runs table order. */
"use strict";

(function () {
  const R = window.RENDER, esc = R.esc, uesc = R.uesc, V = () => window.Verdict;

  // auditor prose and the dissection read go through uesc: a payload that was
  // JSON-encoded twice otherwise prints a backslash-u escape mid sentence
  function para(t) {
    if (t == null || String(t).trim() === "") return `<p class="muted">·</p>`;
    return String(t).trim().split(/\n\s*\n/).map((p) => `<p class="rprose">${uesc(p).replace(/\n/g, "<br>")}</p>`).join("");
  }
  // the agent's own closing word, printed only when it is one of the four
  // recognised outcomes; anything else is left off the card
  const SELF_REPORT = { reproduced: "reproduced", partial: "partial",
    not_reproduced: "not reproduced", unverifiable: "unverifiable" };

  const RunDetail = {
    current: null, view: "agent",

    root() { return document.querySelector("#run-root"); },

    order() {
      const list = window.Runs.ordered();
      const i = list.findIndex((r) => r.id === this.id);
      if (i >= 0) return { list, i };
      const all = window.Data.runs;
      return { list: all, i: all.findIndex((r) => r.id === this.id) };
    },

    navHtml() {
      const { list, i } = this.order();
      const prev = i > 0 ? list[i - 1] : null;
      const next = i >= 0 && i < list.length - 1 ? list[i + 1] : null;
      const btn = (r, label, cls) => r
        ? `<button class="filt ${cls}" data-goto="${esc(r.id)}" title="${esc(r.claim || r.arxiv_id)}">${label}</button>`
        : `<button class="filt ${cls}" disabled>${label}</button>`;
      const pos = i >= 0 ? `<span class="rd-pos tnum">${i + 1} of ${list.length}</span>` : "";
      return `<div class="rd-nav"><button class="crumb" id="rd-back">‹ all runs</button>
        <span class="rd-nav-r">${pos}${btn(prev, "‹ previous", "rd-prev")}${btn(next, "next ›", "rd-next")}</span></div>`;
    },

    auditHtml(run) {
      const a = run.audit || {};
      if (a.score == null && !a.verdict) return "";
      const fam = V().auditFamily(run) || "idle";
      const score = a.score == null ? "·" : a.score;
      const band = a.score >= 8 ? "yes" : a.score >= 6 ? "over" : a.score <= 0 ? "no" : "slate";
      // one flag renderer for the whole site, so this card and the auditor's own
      // verdict JSON below can never print a flag two different ways
      const flags = window.JsonView.flagsHtml(a.flags, "no integrity flags fired on this run");
      const nflags = Array.isArray(a.flags) ? a.flags.length : 0;
      const flagCount = nflags ? `<span class="an-sec-note">${nflags} raised</span>` : "";
      return `<section class="panel-card an-sec rd-audit">
        <div class="pc-head"><span class="plate">audit</span><span class="an-sec-note">${esc(window.Data.auditor.name)}</span></div>
        <div class="rd-audit-head"><span class="jv-score ${band} tnum">${esc(score)}<i>/10</i></span>
          ${V().inline(fam, a.verdict ? String(a.verdict).replace(/_/g, " ") : null)}</div>
        <div class="rd-audit-body"><div class="block-l">rationale</div>${para(a.rationale)}</div>
        <div class="block-l an-flags-l">integrity flags${flagCount}</div>
        <div class="rd-flags">${flags}</div>
      </section>`;
    },

    // the transcript read: one primary failure mode, what went wrong, what the
    // agent did, the quotes that carry it, and the agent's own closing word
    dissectionHtml(run, an, rounds) {
      if (!an || typeof an !== "object" || Array.isArray(an)) return "";
      const mode = window.Modes.get(run.mode);
      const slug = run.mode === "other" && run.mode_slug
        ? `<span class="rd-slug" title="the label recorded for this run">${esc(run.mode_slug)}</span>` : "";
      // a quote only gets the jump link when the round it names has a card below
      const have = new Set((rounds || []).map((r) => String(r.round_index)));
      const roundHtml = (q) => (have.has(String(q.round))
        ? `<button class="an-q-round" data-round="${esc(q.round)}" title="jump to this round in the transcript">round ${esc(q.round)} ↓</button>`
        : `<span class="an-q-round an-q-noround"></span>`);
      const quotes = Array.isArray(an.evidence_quotes) && an.evidence_quotes.length
        ? `<div class="block-l">evidence</div><div class="an-quotes">${an.evidence_quotes.map((q) =>
          `<blockquote class="an-q">${roundHtml(q)}` +
          `<span class="an-q-txt">${uesc(q.quote)}</span></blockquote>`).join("")}</div>` : "";
      const raw = an.self_report != null ? an.self_report : run.self_report;
      const word = SELF_REPORT[String(raw == null ? "" : raw).trim().toLowerCase()] || null;
      const own = word ? `<p class="rd-self"><span class="rd-self-l">Agent's own report:</span> ${esc(word)}</p>` : "";
      return `<section class="panel-card an-sec rd-dissect">
        <div class="pc-head"><span class="plate">dissection</span><span class="an-sec-note">a read of the transcript that never changes the grade</span></div>
        <div class="rd-mode">${window.Modes.chip(run.mode)}${slug}<span class="rd-mode-def">${esc(mode.definition)}</span></div>
        <div class="block-l">what went wrong</div>${para(an.failure_mode_detail)}
        <div class="block-l">what the agent did</div>${para(an.agent_trajectory_summary)}
        ${quotes}${own}
      </section>`;
    },

    transcriptTabsHtml(bundle) {
      const hasAudit = !!(bundle.auditRounds && bundle.auditRounds.length);
      return `<div class="rd-tabs"><span class="plate">transcript</span>
        <span class="rd-tabs-r">
          <button class="filt rd-tab ${this.view === "agent" ? "active" : ""}" data-t="agent">Agent transcript</button>
          ${hasAudit ? `<button class="filt rd-tab ${this.view === "audit" ? "active" : ""}" data-t="audit">Auditor transcript</button>` : ""}
        </span></div>`;
    },

    paintTranscript(bundle) {
      const host = document.querySelector("#rd-transcript");
      if (!host) return;
      const useAudit = this.view === "audit" && bundle.auditRounds && bundle.auditRounds.length;
      const rounds = useAudit ? bundle.auditRounds : bundle.rounds;
      const run = useAudit
        ? Object.assign({}, bundle.run, { budget_h100: null, spent_h100: null, predicted_h100: null })
        : bundle.run;
      R.renderRun(host, run, rounds, { head: false });
    },

    async open(id) {
      this.id = id;
      const host = this.root();
      if (!host) return;
      const run = window.Data.byId[id];
      if (!run) { host.innerHTML = `<div class="empty">That run is not in this collection.</div>`; return; }
      this.view = "agent";
      host.innerHTML = `${this.navHtml()}<div class="run-top rd-head">${R.topHtml(run, { rounds: run.rounds })}</div>
        <div class="empty small">Loading the transcript…</div>`;
      this.wireNav();
      let bundle;
      try { bundle = await window.Data.run(id); }
      catch (e) {
        host.innerHTML = `${this.navHtml()}<div class="run-top rd-head">${R.topHtml(run, { rounds: run.rounds })}</div>` +
          `<div class="empty">The transcript for this run is not published.<br><span class="small">${esc(e.message || String(e))}</span></div>`;
        this.wireNav();
        return;
      }
      if (this.id !== id) return;
      this.current = bundle;
      host.innerHTML = `${this.navHtml()}
        <div class="run-top rd-head">${R.topHtml(bundle.run, { rounds: bundle.rounds.length })}</div>
        ${this.auditHtml(bundle.run)}
        ${this.dissectionHtml(bundle.run, bundle.analysis, bundle.rounds)}
        ${this.transcriptTabsHtml(bundle)}
        <div id="rd-transcript"></div>`;
      this.paintTranscript(bundle);
      this.wireNav();
      this.wireBody(bundle);
    },

    wireNav() {
      const host = this.root();
      const back = host.querySelector("#rd-back");
      if (back) back.onclick = () => window.go(window.runsHash());
      host.querySelectorAll("[data-goto]").forEach((b) => {
        b.onclick = () => window.go("#/run/" + encodeURIComponent(b.dataset.goto));
      });
    },

    wireBody(bundle) {
      const host = this.root();
      host.querySelectorAll(".rd-tab").forEach((b) => {
        b.onclick = () => {
          this.view = b.dataset.t;
          host.querySelectorAll(".rd-tab").forEach((x) => x.classList.toggle("active", x.dataset.t === this.view));
          this.paintTranscript(bundle);
        };
      });
      host.querySelectorAll(".an-q-round").forEach((b) => {
        b.onclick = () => {
          if (this.view !== "agent") {
            this.view = "agent";
            host.querySelectorAll(".rd-tab").forEach((x) => x.classList.toggle("active", x.dataset.t === "agent"));
            this.paintTranscript(bundle);
          }
          // revealRound opens the routine-round group the card may sit inside
          const card = R.revealRound(host, b.dataset.round);
          if (card) {
            card.scrollIntoView({ behavior: "smooth", block: "center" });
            card.classList.add("flash");
            setTimeout(() => card.classList.remove("flash"), 1400);
          }
        };
      });
    },
  };

  window.RunDetail = RunDetail;
})();
