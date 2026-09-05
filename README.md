# RECLAIM

RECLAIM measures whether an autonomous agent can reproduce the central empirical result of a machine learning paper. Each of the 100 papers in the benchmark hands the agent the paper and whatever its authors released. Nothing else is prepared. The agent plans the reproduction, installs the paper's stack, debugs it, and manages its own GPU allocations, then has to recover one pinned target value within a stated numeric tolerance and inside a metered GPU budget capped at 96 H100-hours. A pinned auditor grades the execution evidence the run leaves behind, so a printed number with no trace to a command earns nothing. Papers sit in three tiers according to what the authors released: Run (code, data, and weights all released), Retrain (no released weights, so the model has to be trained first), and Reimplement (no released code either).

## Links

| What | Where |
| --- | --- |
| Hosted trace viewer | https://reclaim-traces.vercel.app |
| Paper | Under review, anonymous |

## Repository layout

| Entry | Contents |
| --- | --- |
| `src/` | The Python packages: `reclaim_repro` (reproduction agent), `reclaim_claude` (pinned grading client), `reclaim_vllm` (audit schema, rubric finalizer, shared runtime), `reclaim_serve` (serving launcher), `reclaim_data`, `reclaim_openai`. `src/run_arxiv_prompt_vllm.py` is the vLLM-backed auditor entry point the cluster scripts call. |
| `tests/` | Test suite for the packages above. |
| `prompts/` | The reproduction prompt, the audit prompt, the two artifact-availability classifier prompts used to build the splits (`prompt.txt` and `prompt_openai_websearch.txt`), and one rendered example. |
| `rubric_audit.md` | The frozen audit rubric the grader reads. |
| `scripts/` | Serve, reproduce, and cluster scripts. Site-specific values are placeholders. See `scripts/README.md`. |
| `data/` | `data/splits/` holds the two frozen split files. `data/LICENSE` covers them. |
| `runs/` | The run records: `index.json` plus one gzipped bundle per run. |
| `viewer/` | Static browser viewer for the run records. The hosted copy is linked above. |
| `requirements.txt` | Core dependencies. |
| `requirements-serve.txt` | Adds vLLM. Needed only to serve an open-weight model locally. |
| `ruff.toml` | Lint configuration. |

## Install

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To serve an open-weight agent model on your own GPUs, also install the serving extras.

```bash
pip install -r requirements-serve.txt
```

Two environment variables are optional. `RECLAIM_TELEMETRY_URL` and `RECLAIM_TELEMETRY_KEY` point the harness at a run database. Leave them unset and nothing is uploaded.

## Quickstart

The three commands below serve a model, run one paper, and grade one run.

```bash
# Serve the agent model and publish the address other nodes reach it at.
$ PYTHONPATH=src python3 -m reclaim_serve \
    --model <hf-id-or-path> \
    --served-model-name <model-id> \
    --port 8000 \
    --tensor-parallel-size <tp> \
    --endpoint-file <endpoint-file>
$ export RECLAIM_ENDPOINT_FILE=<endpoint-file>

# Run one paper of the frozen benchmark against it.
$ PYTHONPATH=src python3 -m reclaim_repro \
    --paper-id <arxiv-id> \
    --split eval \
    --run-id <run-id> \
    --runs-dir <runs-dir> \
    --vllm-server-url <endpoint-url> \
    --served-model-name <model-id>

# Hosted model: same run command, its API as the server URL, key and window exported.
$ export RECLAIM_API_KEY=<api-key> RECLAIM_CONTEXT_LENGTH=<window>
$ PYTHONPATH=src python3 -m reclaim_repro ... \
    --vllm-server-url https://api.meta.ai/v1 --served-model-name <model-id>

# Grade the bundle with the pinned grader (hosted sweeps: --batch <model>-<timestamp>).
$ PYTHONPATH=src python3 -m reclaim_claude \
    --batch slurm-<jobid> \
    --runs-dir <runs-dir> \
    --split eval \
    --model claude-sonnet-5 \
    --extracted-output <verdicts.jsonl>
```

The grading command takes a whole batch at once. It grades each paper's newest attempt and skips runs still in progress unless `--include-running` is passed, since a half-written bundle measures the harness.

### Reading the verdict file

`--extracted-output <verdicts.jsonl>` writes one JSON object per graded run. The auditor returns the first group of fields. The rest are derived in code by `finalize_audit_row`, so a verdict can be rechecked from the row without rerunning the grader.

| Field | Meaning |
| --- | --- |
| `custom_id`, `paper_id` | The paper's arXiv ID. |
| `central_claim` | The claim under grade. |
| `match_bar_kind`, `target_metric`, `target_scope`, `reference_value`, `op`, `tolerance` | The pinned target the run is graded against. The auditor adopts the dataset row's tuple and sets only the comparison operator and the tolerance. |
| `execution_verified`, `execution_evidence` | Whether the run computed the metric, and the cited proof. |
| `measured_value`, `measured_citation` | The value the run produced and the file and line it came from. |
| `cheat_flags` | Typed provenance breaks. Each carries a `kind`, cited `evidence`, and a `severity` of `low`, `med`, or `high`. |
| `value_comparison`, `methodology_notes` | The comparison against the bar, and fidelity notes. |
| `score` | The auditor's integer reproduction score. |
| `confidence`, `rationale` | The auditor's own confidence and its one-paragraph justification. |
| `has_high_cheat_flag` | Set when any flag is high severity. A high-severity flag caps the score. |
| `reported_score` | Present only when a cap lowered the score. It holds the score the auditor assigned. |
| `verdict` | Derived from the score. See below. |
| `reproduced` | Boolean shorthand for a `reproduced` verdict. |
| `verification_status`, `exit_reason` | Whether the grading pass itself completed, and how the grader's loop ended. |

`verdict` takes one of five values.

| Verdict | When |
| --- | --- |
| `reproduced` | The score sits in the top band. |
| `partial` | The run executed a faithful pipeline and landed short of the bar. |
| `not_reproduced` | The score sits below the partial band and the run is not disqualified. |
| `unverifiable` | The lowest non-zero score together with an auditor report that execution was never verified. |
| `disqualified` | A high-severity provenance flag, or a score of zero. |

## Dataset

Two frozen files, both JSON Lines, one paper per line.

| File | Split | Papers |
| --- | --- | --- |
| `data/splits/eval_100.jsonl` | Evaluation | 100 |
| `data/splits/dev_14.jsonl` | Development | 14 |

### Row fields

| Field | Meaning |
| --- | --- |
| `custom_id` | The paper's arXiv ID. Primary key. |
| `central_claim` | The paper's central empirical claim in one sentence. |
| `claim_evidence` | Quoted paper text that grounds the claim. |
| `paper_kind` | Paper type. Every row is `empirical`. |
| `mre_config` | Prose recipe for the cheapest configuration that still tests the claim. |
| `agent_task` | The reproduction instruction handed to the agent. |
| `verified_links` | Artifact URLs confirmed by a tool call, under `paper_or_project`, `code`, `dataset`, and `weights`. Each is a list and is empty when nothing of that kind was located. |
| `signals` | The per-artifact availability audit, with one object for `code_available`, `dataset_available`, `weights_available`, and `dataset_is_standard`. Each carries a `value`, a `verification` state, and the tool observation as `evidence`. |
| `match_target` | The quantity a run has to recover. Described below. |
| `score` | Artifact-availability score derived from the four signals. |
| `tier` | The row's artifact tier. `Easy`, `Medium`, and `Hard` in the file are the Run, Retrain, and Reimplement tiers. |
| `split` | `eval` for the evaluation split, `dev` for the development split. |
| `h100_estimate` | The compute estimate that placed the paper in its band: `hours`, `basis_kind`, the reported `gpu_count` and `gpu_type`, `wallclock_hours`, the `h100_equivalent_multiplier`, and a free-text `basis`. |
| `h100_hours_estimate`, `h100_estimate_basis` | The estimated hours and its justification, flattened out of `h100_estimate`. |
| `audited_h100_hours`, `h100_recomputed_hours`, `h100_arithmetic_mismatch`, `h100_hours_adjudicated`, `h100_needs_human_review` | The estimate after the arithmetic audit, the recomputed value, and the flags recording whether the recompute replaced the stated estimate or the row went to human review. |
| `h100_band`, `selection_band` | The compute band used for stratified selection. The two agree on every row. |
| `verification_status`, `web_verification`, `exit_reason` | Classifier states carried over from dataset construction. |

### Match target and tolerance

`match_target` is the object a run is graded against. It holds `config` (the experimental configuration that produces the value), `metric` (the reported metric name), `value` (the target value), `scope` (the model, dataset, or setting the value applies to), and `match_bar_kind` (the shape of the bar). The released rows use `point_estimate`, `direction`, `threshold`, `magnitude`, and `range`.

The agent receives the metric, the target value, the scope, and the bar shape. It does not receive `config`, so it has to design the experiment itself.

The tolerance is the width of the bar. For a point estimate a run matches when its measured value lands within the tolerance the authors state, or within the default relative tolerance when the paper states none. `rubric_audit.md` carries the default and the operator each bar kind uses. A threshold is a floor or a ceiling the run has to clear. A direction target is a baseline the run has to beat. A magnitude target grades the size of a reported delta, and a range target grades whether the value falls inside a reported interval.

### Loading the splits

The harness reads `data/splits/` by default. `RECLAIM_SPLITS_DIR` points it at another directory. Passing `--lockfile <owner/name>` loads a Hugging Face dataset by name instead of the local files.

## Run records

`runs/` holds 372 graded runs, four agents across the three tiers. The agents are DeepSeek-V4-Flash, Qwen3.6-27B, MiniMax-M2.7, and Muse Spark 1.2. Every run was graded by the pinned auditor, Claude Sonnet 5.

```
runs/
  index.json                 summary of every run, plus the papers, models, tiers, and mechanisms
  <run-id>.json.gz           one gzipped bundle per run
```

A run id is `<model>-<tier>-<arxiv-id>`, for example `dsv4-reimplement-2402.04579`.

### `index.json`

| Key | Contents |
| --- | --- |
| `generated` | The date the index was built. |
| `benchmark` | Benchmark name, paper count, and venue. |
| `auditor` | The pinned grader's id and display name. |
| `models` | One entry per agent, with its key, display name, and served model id. |
| `tiers` | The three tiers, each with its key, name, and what the tier releases. |
| `modes` | The failure mechanisms, each with its key and display name. |
| `sweeps` | One entry per model and tier pair, with run count, verdict counts, mechanism counts, the score distribution, and the compute spent against the compute granted. |
| `papers` | One entry per benchmark paper, with its tier, band, claim, target, artifact URLs, predicted compute, and a short gist. |
| `runs` | One summary entry per run, in the shape below. |

### Run fields

The `runs` entries of `index.json` and the `run` object inside each bundle carry the same fields.

| Field | Meaning |
| --- | --- |
| `id` | The run id, which is also the bundle file name. |
| `arxiv_id` | The paper the run attempted. |
| `model`, `tier`, `sweep` | The agent, the tier, and the model and tier pair the run belongs to. |
| `exit_label` | How the episode ended: `Finished`, `Budget exhausted`, or `Context limit`. |
| `rounds`, `tool_calls`, `duration_s` | Agent rounds used, tool calls issued, and wall-clock seconds. |
| `budget_h100`, `spent_h100` | The compute granted and the compute charged, in H100-equivalent hours. |
| `tokens` | Token counts for the episode: `prompt`, `completion`, `total`, `cached`, and `reasoning`. |
| `audit` | The pinned grade: `score`, `verdict`, `reproduced`, `flags`, `rationale`, and `has_transcript`. |
| `mode` | The run's failure mechanism under the nine names below, or `other` when the dissection label falls outside them. |
| `mode_slug` | The raw label the dissection wrote. It includes retired names and underscore spellings, which `mode` normalizes. |
| `claim` | The paper's central claim. |
| `target` | The pinned match target in one line. |
| `self_report` | The agent's own verdict on its run. It is recorded and never treated as ground truth. |

### Bundle contents

Each `runs/<id>.json.gz` decompresses to a JSON object with four keys.

| Key | Contents |
| --- | --- |
| `run` | The run fields above. |
| `events` | The ordered event transcript: agent turns with their reasoning, tool calls with arguments, and tool results with exit codes, output, duration, and compute charged. |
| `analysis` | The transcript dissection: `paper_gist`, `failure_mode_detail`, `agent_trajectory_summary`, `evidence_quotes`, and `self_report`. |
| `audit_events` | The grader's own tool loop over the run bundle. |

### Failure mechanisms

The dissection assigns exactly one primary mechanism per run. It runs after the audit and separately from it, and it never moves a grade.

| Mechanism | Definition |
| --- | --- |
| `reproduced-clean` | The agent runs the authors' released artifact on the authors' data and the graded quantity lands inside the pinned bar. |
| `near-miss-partial` | The agent executes a faithful pipeline and produces a measured number that falls short of the bar. |
| `reimplement-without-validating` | The agent writes its own version of the method or one of its components and anchors no stage of it to a reference it did not produce itself. |
| `environment-fights` | Build and dependency failures in the released stack consume the decisive rounds, and the claim's experiment never reaches a valid number. |
| `artifact-provenance-mismatch` | The agent measures a checkpoint, dataset, split, or scale other than the one the dataset row names. |
| `scope-substitution` | The experiment that ran differs from the experiment the pinned claim is about. |
| `stale-artifact-reliance` | The agent runs the right thing, gets a number that misses, finds a shipped constant in the repository that passes, and reports the constant. |
| `procrastination/wall-kill` | The agent spends its rounds on reading, retrieval, and probing, the pinned pipeline never runs, and the run stops with most of its allocation unspent. |
| `killed-before-the-number` | The environment and the artifacts work, and the run ends before the graded command yields a value. |

### Opening one bundle

```python
import gzip
import json

with gzip.open("runs/dsv4-reimplement-2402.04579.json.gz", "rt") as handle:
    bundle = json.load(handle)

run = bundle["run"]
print(run["audit"]["score"], run["audit"]["verdict"])
print(run["mode"], run["mode_slug"])
print(run["rounds"])
```

That prints the auditor's score and the verdict derived from it, then the run's failure mechanism together with the raw dissection label, then how many rounds the agent used.


## Running on a cluster

The scripts under `scripts/` assume a Slurm scheduler and an Apptainer container runtime. The serve scripts start a vLLM server on allocated GPUs and write the endpoint address to a file the run command reads. The reproduce scripts launch one batch job per model per tier, then run that tier's papers as processes against that server. They are named for the tier they run: `scripts/reproduce/<agent>/run_<agent>.sbatch`, `retrain_<agent>.sbatch`, and `reimplement_<agent>.sbatch`, with `scripts/reproduce/muse_spark/run_muse_spark.sh`, `retrain_muse_spark.sh`, and `reimplement_muse_spark.sh` for the hosted API. Every agent GPU step goes through `run_gpu`, which holds its own Slurm allocation and charges the time against the paper's metered compute budget, so serving GPUs and agent GPUs are accounted separately. `scripts/cluster/build_cuda_sandbox.sh` builds the container image the agent's shell steps run inside.

Site-specific values are placeholders. Fill them in before running anything.

| Placeholder | Value to supply |
| --- | --- |
| `<account>` | Slurm charge account. |
| `<partition>` | GPU partition for serving and for agent allocations. |
| `<scratch>` | Scratch filesystem root for run bundles and workspaces. |
| `<home>` | Home directory root. |
| `<sif>` | Path to the built Apptainer image. |
| `<module>` | Environment module to load. |
| `<hf-cache>` | Shared Hugging Face cache directory. |
| `<venv>` | Path to the virtual environment. |

`scripts/README.md` lists the same placeholders per script. The account, partition, and scratch root can also come from `RECLAIM_SLURM_ACCOUNT`, `RECLAIM_SLURM_PARTITION`, and `RECLAIM_SCRATCH`. Run `python3 -m reclaim_repro --help` for the full set and the current defaults.

A tier job runs its papers in ascending order of compute ceiling, so the cheapest papers finish first. Set `RESUME=1` to restart a killed job. It skips papers whose verdict is already on disk, regrades nothing, and retries any paper whose run ended without a verdict.

## Anonymity note

The run records were passed through a text scrub before release. It rewrites hostnames and node names, filesystem paths, usernames and account names, scheduler job and step ids, API keys and tokens, email addresses, IP addresses, wall-clock timestamps, and identifiers that would name the hardware or the site. A small number of dissection notes were withheld for the same reason. The records are otherwise unchanged: no score, verdict, flag, or transcript event was edited. The dataset identifier the splits publish under is masked for review, so load the local files under `data/splits/` instead of the `--lockfile` path.

The frozen prompts and the rubric are printed exactly as the sweeps used them, so they describe the authors' deployment (a CPU orchestration host, node-local `/tmp`) and they call the grading pass "Stage-7", the internal name for the auditor. Four run transcripts contained `ps aux` output listing other users' project directories on the shared cluster, and those paths were replaced with `/projects/[proj]/[user]` in this release.

## License

The code in this repository is released under the MIT License. See `LICENSE`.

The splits under `data/splits/` and the run records under `runs/` are released under CC BY 4.0. See `data/LICENSE`.

Both are attributed to RECLAIM Authors. The papers the benchmark points at remain under their own licenses.

## Citation

```bibtex
@misc{reclaim2026,
  title  = {RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?},
  author = {RECLAIM Authors},
  year   = {2026},
  note   = {Under review}
}
```
