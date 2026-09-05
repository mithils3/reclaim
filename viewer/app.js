/* app.js: boot, hash routing and the global filter bar. Five views over one
   static index: OVERVIEW (the landing worksheet), RUNS (the filtered table),
   RUN (one transcript), PAPERS (the per-agent outcome grid) and ABOUT. The
   Agent and Tier selects live in the top bar and hold across Overview, Runs and
   Papers; Runs adds verdict, failure mode and a text search on top of them. */
"use strict";

const $ = (s) => document.querySelector(s);

const State = {
  model: "all", tier: "all", verdict: "all", mode: "all", q: "",
  view: "overview", runId: null, arxiv: null,
};
window.State = State;

const VIEWS = ["overview", "runs", "run", "papers", "about"];
const FILTERED_VIEWS = { overview: 1, runs: 1, papers: 1 };

// ---- hash routing ----------------------------------------------------------
function parseHash() {
  const raw = location.hash.replace(/^#/, "") || "/overview";
  const qi = raw.indexOf("?");
  const path = qi >= 0 ? raw.slice(0, qi) : raw;
  const params = new URLSearchParams(qi >= 0 ? raw.slice(qi + 1) : "");
  return { parts: path.split("/").filter(Boolean), params };
}
function runsHash(over) {
  const o = Object.assign({ model: State.model, tier: State.tier, verdict: State.verdict, mode: State.mode, q: State.q }, over || {});
  const p = new URLSearchParams();
  ["model", "tier", "verdict", "mode"].forEach((k) => { if (o[k] && o[k] !== "all") p.set(k, o[k]); });
  if (o.q) p.set("q", o.q);
  const s = p.toString();
  return "#/runs" + (s ? "?" + s : "");
}
function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }
window.go = go;
window.runsHash = runsHash;

function showView(v) {
  State.view = v;
  VIEWS.forEach((name) => {
    const elx = $("#view-" + name);
    if (elx) elx.classList.toggle("hidden", name !== v);
  });
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === v || (v === "run" && t.dataset.view === "runs") || (v === "paper" && t.dataset.view === "papers");
    t.classList.toggle("active", on);
  });
  // About has nothing to filter, but taking the bar out of the page dropped the
  // heading 44px on the way in and lifted it again on the way out. The bar holds
  // its height there and states the collection instead of carrying controls.
  const bar = $("#filterbar"), stat = v === "about";
  if (bar) {
    bar.classList.toggle("hidden", !FILTERED_VIEWS[v] && !stat);
    bar.classList.toggle("fb-static", stat);
  }
}

function route() {
  const { parts, params } = parseHash();
  const head = parts[0] || "overview";
  if (head === "runs") {
    ["model", "tier", "verdict", "mode"].forEach((k) => { State[k] = params.get(k) || "all"; });
    State.q = params.get("q") || "";
    syncBar();
    showView("runs");
    window.Runs.render();
  } else if (head === "run" && parts[1]) {
    State.runId = decodeURIComponent(parts[1]);
    showView("run");
    window.RunDetail.open(State.runId);
  } else if (head === "papers") {
    showView("papers");
    window.PapersView.renderList();
  } else if (head === "paper" && parts[1]) {
    State.arxiv = decodeURIComponent(parts[1]);
    showView("papers");
    window.PapersView.renderPaper(State.arxiv);
  } else if (head === "about") {
    showView("about");
  } else {
    showView("overview");
    window.OverviewView.render();
  }
  const main = document.querySelector("#view-" + (State.view === "paper" ? "papers" : State.view));
  if (main) main.scrollTop = 0;
}
window.route = route;

// ---- global filter bar -----------------------------------------------------
function buildBar() {
  const D = window.Data;
  const opt = (v, l, cur) => `<option value="${window.RENDER.esc(v)}"${v === cur ? " selected" : ""}>${window.RENDER.esc(l)}</option>`;
  const agents = [opt("all", "all agents", State.model)].concat(D.models.map((m) => opt(m.key, m.name, State.model))).join("");
  const tiers = [opt("all", "all tiers", State.tier)].concat(D.tiers.map((t) => opt(t.key, t.name, State.tier))).join("");
  const papers = D.benchmark.papers != null ? D.benchmark.papers : D.papers.length;
  $("#filterbar").innerHTML =
    `<label class="fb-l" for="fb-model">Agent</label><select id="fb-model" class="fb-sel">${agents}</select>` +
    `<label class="fb-l" for="fb-tier">Tier</label><select id="fb-tier" class="fb-sel">${tiers}</select>` +
    `<span class="fb-what" id="fb-what"></span>` +
    `<button class="filt fb-reset" id="fb-reset" type="button">reset</button>` +
    // the line the bar shows on About, where there is nothing to filter
    `<span class="fb-note">all ${D.runs.length} runs · ${papers} papers</span>`;
  $("#fb-model").onchange = (e) => { State.model = e.target.value; onBarChange(); };
  $("#fb-tier").onchange = (e) => { State.tier = e.target.value; onBarChange(); };
  $("#fb-reset").onclick = () => { State.model = "all"; State.tier = "all"; State.verdict = "all"; State.mode = "all"; State.q = ""; onBarChange(); };
  syncBar();
}
// the selected tier's one-line meaning rides next to the selects
function syncBar() {
  const m = $("#fb-model"), t = $("#fb-tier"), w = $("#fb-what");
  if (m) m.value = State.model;
  if (t) t.value = State.tier;
  if (w) {
    const tier = window.Data.tier(State.tier);
    w.textContent = State.tier === "all" ? "" : (tier.what || "");
  }
}
function onBarChange() {
  syncBar();
  if (State.view === "runs") { go(runsHash()); return; }
  if (State.view === "papers") { window.PapersView.renderList(); return; }
  window.OverviewView.render();
}

// ---- about page: the few numbers and names that come from the data ---------
function fillAbout() {
  const D = window.Data, esc = window.RENDER.esc;
  const set = (key, text) => document.querySelectorAll(`[data-b="${key}"]`).forEach((n) => { n.textContent = text; });
  if (D.benchmark.papers != null) set("papers", String(D.benchmark.papers));
  if (D.benchmark.venue) set("venue", D.benchmark.venue);
  if (D.benchmark.name) set("name", D.benchmark.name);
  set("auditor", D.auditor.name);
  const host = $("#about-tiers");
  if (host && D.tiers.length) {
    host.innerHTML = D.tiers.map((t) => `<li><b>${esc(t.name)}.</b> ${esc(t.what)}</li>`).join("");
  }
}

// ---- boot ------------------------------------------------------------------
async function boot() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => go(t.dataset.view === "runs" ? runsHash() : "#/" + t.dataset.view));
  });
  try {
    await window.Data.load();
  } catch (e) {
    document.querySelectorAll("main.view").forEach((m) => m.classList.add("hidden"));
    const host = $("#view-overview");
    host.classList.remove("hidden");
    host.innerHTML = `<section class="detail"><div class="empty">The run data is not published alongside this page yet.<br><span class="small">${window.RENDER.esc(e.message || String(e))}</span></div></section>`;
    return;
  }
  buildBar();
  fillAbout();
  const count = $("#brand-count");
  if (count) {
    const papers = window.Data.benchmark.papers != null ? window.Data.benchmark.papers : window.Data.papers.length;
    count.textContent = `${window.Data.runs.length} runs · ${papers} papers`;
  }
  window.addEventListener("hashchange", route);
  route();
}
document.addEventListener("DOMContentLoaded", boot);
