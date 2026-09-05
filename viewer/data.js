/* data.js: the static data layer. Reads a pre-built index plus one gzipped
   bundle per run out of a folder of flat files; there is no live connection and
   nothing is written back. The folder defaults to "../runs/" and can be pointed
   elsewhere with a ?data= query parameter. It also holds the two fixed
   vocabularies the whole site reads from: the tier set (Run, Retrain,
   Reimplement) and the failure-mode set, whose order, display name, colour and
   one-line definition never vary between views. rowsToRounds() reconstructs the
   Round/Call shape render.js draws, so the same renderer serves both the agent
   transcript and the auditor transcript. */
"use strict";

(function () {
  function basePath() {
    const m = /[?&]data=([^&]*)/.exec(location.search);
    let b = m ? decodeURIComponent(m[1]) : "../runs/";
    if (!b) b = "../runs/";
    return b.endsWith("/") ? b : b + "/";
  }
  const BASE = basePath();

  // ---- failure-mode vocabulary ----------------------------------------------
  // index.modes fixes the order and the display names. This table fixes the
  // colour every view paints a mode with, and the one-line definition the run
  // page prints beside it. Other is always last.
  const MODE_ORDER = [
    "reproduced-clean", "near-miss-partial", "reimplement-without-validating",
    "environment-fights", "artifact-provenance-mismatch", "scope-substitution",
    "stale-artifact-reliance", "procrastination/wall-kill",
    "killed-before-the-number", "other",
  ];
  // ten hues spread around the wheel so they stay apart at a small swatch: a 10px
  // legend square and a stacked bar segment
  const MODE_COLOUR = {
    "reproduced-clean": "--mode-reproduced",
    "near-miss-partial": "--mode-nearmiss",
    "reimplement-without-validating": "--mode-reimpl",
    "environment-fights": "--mode-envfight",
    "artifact-provenance-mismatch": "--mode-provenance",
    "scope-substitution": "--mode-scope",
    "stale-artifact-reliance": "--mode-stale",
    "procrastination/wall-kill": "--mode-procrast",
    "killed-before-the-number": "--mode-killed",
    "other": "--mode-other",
  };
  const MODE_DEF = {
    "reproduced-clean": "The run measured the claim's number itself and the evidence carries it.",
    "near-miss-partial": "The pipeline ran and produced a real number that falls short of the claim's bar or its scope.",
    "reimplement-without-validating": "The agent rebuilt the method and reported from it without checking the rebuild against anything.",
    "environment-fights": "The attempt went into getting the code, the data or the dependencies to run at all.",
    "artifact-provenance-mismatch": "The reported number does not come from the artifact the run points at.",
    "scope-substitution": "The run measured a smaller or different setting than the claim names and reported it as the claim.",
    "stale-artifact-reliance": "The run leaned on an earlier or partial artifact instead of producing the measurement.",
    "procrastination/wall-kill": "The decisive run was deferred until the budget or the round limit ended the attempt.",
    "killed-before-the-number": "The run reached a limit before any measurement of the claim existed.",
    "other": "A mode outside the fixed vocabulary. The recorded label is shown beside it.",
  };
  const titleCase = (k) => String(k).split(/[-/_]/).filter(Boolean)
    .map((w, i) => (i ? w : w.charAt(0).toUpperCase() + w.slice(1))).join(" ");

  const Modes = {
    all: [], map: {},

    hydrate(list) {
      const rows = (Array.isArray(list) && list.length)
        ? list.slice()
        : MODE_ORDER.map((k) => ({ key: k, name: titleCase(k) }));
      rows.sort((a, b) => (a.key === "other" ? 1 : 0) - (b.key === "other" ? 1 : 0));
      this.all = rows.map((m) => ({
        key: m.key,
        name: m.name || titleCase(m.key),
        colour: MODE_COLOUR[m.key] || "--slate",
        definition: MODE_DEF[m.key] || "",
      }));
      this.map = {};
      this.all.forEach((m, i) => { m.rank = i; this.map[m.key] = m; });
      return this.all;
    },

    get(key) {
      return this.map[key] || { key, name: key ? titleCase(key) : "not recorded",
        colour: "--slate", definition: "", rank: 99 };
    },
    name(key) { return this.get(key).name; },
    definition(key) { return this.get(key).definition; },
    rank(key) { return this.get(key).rank; },
    // the mode pill: a fixed colour dot plus the display name, same everywhere
    chip(key, extra) {
      const m = this.get(key);
      return `<span class="mchip ${extra || ""}" style="--mc:var(${m.colour})">` +
        `<span class="mchip-t">${window.RENDER.esc(m.name)}</span></span>`;
    },
  };

  const Data = {
    index: null, benchmark: {}, models: [], tiers: [], sweeps: [], papers: [], runs: [],
    byId: {}, byModel: {}, byPaper: {}, tierByKey: {}, modelName: {}, base: BASE,
    _runCache: new Map(), _loading: null,

    async load() {
      if (this.index) return this.index;
      if (this._loading) return this._loading;
      this._loading = (async () => {
        const res = await fetch(BASE + "index.json", { cache: "no-cache" });
        if (!res.ok) throw new Error("index not available (" + res.status + ")");
        const ix = await res.json();
        this.hydrate(ix);
        return ix;
      })();
      return this._loading;
    },

    hydrate(ix) {
      this.index = ix;
      this.benchmark = ix.benchmark || {};
      this.models = ix.models || [];
      this.tiers = ix.tiers || [];
      this.sweeps = ix.sweeps || [];
      this.papers = ix.papers || [];
      this.auditor = ix.auditor || { id: "auditor", name: "Auditor" };
      Modes.hydrate(ix.modes);
      this.tierByKey = {};
      this.tiers.forEach((t) => { this.tierByKey[t.key] = t; });
      this.modelName = {};
      this.models.forEach((m) => { this.modelName[m.key] = m.name; });
      this.byPaper = {};
      this.papers.forEach((p) => { this.byPaper[p.arxiv_id] = p; });
      this.runs = (ix.runs || []).map((r) => this.decorate(r));
      this.runs.sort((a, b) =>
        this.tierRank(a.tier) - this.tierRank(b.tier) ||
        String(a.model).localeCompare(String(b.model)) ||
        String(a.arxiv_id).localeCompare(String(b.arxiv_id)));
      this.byId = {};
      this.runs.forEach((r) => { this.byId[r.id] = r; });
      this.byModel = {};
      this.runs.forEach((r) => { (this.byModel[r.model] || (this.byModel[r.model] = [])).push(r); });
      return this;
    },

    tier(key) { return this.tierByKey[key] || { key, name: key || "", what: "" }; },
    tierName(key) { return this.tier(key).name || key || ""; },
    tierRank(key) { const i = this.tiers.findIndex((t) => t.key === key); return i < 0 ? 9 : i; },
    modelLabel(key) { return this.modelName[key] || key || ""; },

    // flatten the parts every renderer reads off the run row itself
    decorate(r) {
      const a = r.audit || {};
      const p = this.byPaper[r.arxiv_id] || null;
      const m = this.models.find((x) => x.key === r.model) || null;
      return Object.assign({}, r, {
        paper: p,
        predicted_h100: p && p.predicted_h100 != null ? p.predicted_h100 : null,
        band: p ? p.band : null,
        model_name: m ? m.name : r.model,
        model_id: m ? m.id : null,
        audit_verdict: a.verdict || null,
        audit_score: a.score != null ? a.score : null,
        audit_reproduced: a.reproduced != null ? a.reproduced : null,
        mode_name: window.Modes.name(r.mode),
        claim: r.claim || (p ? p.claim : null),
      });
    },

    runsForPaper(arx) { return this.runs.filter((r) => r.arxiv_id === arx); },

    // ---- one run bundle: <id>.json.gz ---------------------------------------
    async run(id) {
      if (this._runCache.has(id)) return this._runCache.get(id);
      const promise = (async () => {
        const res = await fetch(BASE + encodeURIComponent(id) + ".json.gz", { cache: "no-cache" });
        if (!res.ok) throw new Error("run bundle not available (" + res.status + ")");
        const buf = new Uint8Array(await res.arrayBuffer());
        let text;
        if (buf[0] === 0x1f && buf[1] === 0x8b) {
          if (typeof DecompressionStream !== "function") throw new Error("this browser cannot decompress the bundle");
          const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
          text = await new Response(stream).text();
        } else {
          text = new TextDecoder().decode(buf);
        }
        const bundle = JSON.parse(text);
        bundle.run = Object.assign({}, this.byId[id] || {}, this.decorate(bundle.run || {}));
        bundle.rounds = this.rowsToRounds(bundle.events || []);
        bundle.auditRounds = bundle.audit_events ? this.rowsToRounds(bundle.audit_events) : null;
        return bundle;
      })();
      this._runCache.set(id, promise);
      promise.catch(() => this._runCache.delete(id));
      return promise;
    },

    rowsToRounds(rows) {
      const byKey = new Map(), order = [];
      const ensure = (kind, idx) => {
        const key = kind + ":" + idx;
        let r = byKey.get(key);
        if (!r) {
          r = { round_index: idx, kind, arxiv_id: null, ts: null, exit_reason: null,
            finish_reason: null, reasoning: "", content: "", calls: [] };
          byKey.set(key, r); order.push(r);
        }
        return r;
      };
      for (const e of rows) {
        if (e.kind === "round_open" || e.kind === "final") {
          const r = ensure(e.kind === "final" ? "final" : "round", e.round_index);
          if (e.reasoning) r.reasoning = e.reasoning;
          if (e.content) r.content = e.content;
          if (e.finish_reason) r.finish_reason = e.finish_reason;
          if (e.kind === "final") r.exit_reason = e.exit_reason || r.exit_reason;
        } else if (e.kind === "call_start") {
          const r = ensure("round", e.round_index);
          r.calls.push({ tool_name: e.tool_name, command: e.command, detail_kind: e.detail_kind,
            args: e.args, stdout: "", stderr: "", truncated: false });
        } else if (e.kind === "call_result") {
          const r = ensure("round", e.round_index);
          let c = r.calls[r.calls.length - 1];
          if (!c) { c = { tool_name: e.tool_name || "?", stdout: "", stderr: "", truncated: false }; r.calls.push(c); }
          c.ok = e.ok; c.rc = e.rc; c.duration_s = e.duration_s;
          c.cost_h100 = e.cost_h100; c.remaining_h100 = e.remaining_h100;
          c.error = e.error; c.path = e.path;
          c.stdout = e.stdout || ""; c.stderr = e.stderr || ""; c.truncated = !!e.truncated;
        }
      }
      return order;
    },
  };

  window.Data = Data;
  window.Modes = Modes;
})();
