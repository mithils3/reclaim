/* verdict.js: the single source of truth that collapses an auditor verdict into
   FOUR semantic verdict families, each with a glyph, a word and a colour. Every
   view reads verdicts from here so a run looks the same in the runs table, the
   paper page, the transcript and the trace cap. Also home to the rotated stamp.
     reproduced ● green   miss ◐ amber   fault ✕ coral   idle ○ slate */
"use strict";

(function () {
  const esc = window.RENDER.esc;

  const FAMILY = {
    reproduced: { glyph: "●", cssvar: "--yes", word: "reproduced", stamp: "REPRODUCED" },
    miss:       { glyph: "◐", cssvar: "--over", word: "miss",       stamp: "PARTIAL" },
    fault:      { glyph: "✕", cssvar: "--no",  word: "fault",       stamp: "NOT REPRODUCED" },
    idle:       { glyph: "○", cssvar: "--slate", word: "idle",      stamp: "NEVER RAN" },
    done:       { glyph: "◌", cssvar: "--slate", word: "unjudged",  stamp: "UNJUDGED" },
  };

  // Auditor verdict -> family. The auditor's vocabulary on the 0 to 10 scale:
  // disqualified (0), reproduced (8 to 10), partial (6 to 7), unverifiable (never
  // executed), not_reproduced (everything else).
  const AUDIT_FAMILY = {
    reproduced: "reproduced", partial: "miss",
    not_reproduced: "fault", unverifiable: "fault", disqualified: "fault",
  };
  function familyFromAudit(run) {
    if (!run) return null;
    const v = run.audit_verdict || (run.audit && run.audit.verdict);
    if (v) return AUDIT_FAMILY[String(v).toLowerCase().trim()] || null;
    const rep = run.audit_reproduced != null ? run.audit_reproduced : (run.audit && run.audit.reproduced);
    if (rep === true) return "reproduced";
    if (rep === false) return "fault";
    return null;
  }

  const Verdict = {
    FAMILY,
    auditFamily(run) { return familyFromAudit(run); },

    // a single run -> family key: every published run carries a grade, so the
    // auditor verdict decides and idle is only the no-grade fallback
    ofRun(run) {
      if (!run) return "idle";
      return familyFromAudit(run) || "idle";
    },

    // a paper -> family: reproduced if any run is; else closest-to-success of its runs
    ofPaper(reproduced, runFamilies) {
      if (reproduced) return "reproduced";
      const set = new Set(runFamilies);
      for (const f of ["miss", "fault", "idle", "done"]) if (set.has(f)) return f;
      return "idle";
    },

    meta(fam) { return FAMILY[fam] || FAMILY.idle; },

    // inline verdict: glyph + word (word overridable to show the specific verdict)
    inline(fam, wordOverride) {
      const m = this.meta(fam);
      const w = wordOverride || m.word;
      return `<span class="vd ${fam}"><span class="vg">${m.glyph}</span>${esc(w)}</span>`;
    },

    // the most specific human word for a run (the auditor verdict wins)
    word(run) {
      const fam = this.ofRun(run);
      const v = run && (run.audit_verdict || (run.audit && run.audit.verdict));
      if (v && familyFromAudit(run) === fam) return String(v).replace(/_/g, " ");
      return this.meta(fam).word;
    },

    // rotated double-entry stamp
    stamp(fam, labelOverride) {
      const m = this.meta(fam);
      return `<span class="stamp ${fam}">${esc(labelOverride || m.stamp)}</span>`;
    },
  };

  window.Verdict = Verdict;
})();
