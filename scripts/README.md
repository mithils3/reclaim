# Batch scripts

Every script here is a launcher. The reproduction agent, the auditor, and the serving
layer all live under `src/`; these files only decide what runs, where, and against
which model. Fill in the bracketed placeholders before the first run.

## Scripts

| Path | What it does | Model and tier |
| --- | --- | --- |
| `cluster/build_cuda_sandbox.sh` | Builds the Apptainer image every agent shell step runs inside. | any |
| `serve/serve_gh200.sbatch` | Serves one model with vLLM on a single node and publishes its URL to a shared file. | any |
| `serve/serve_multinode.sbatch` | Serves one model across several nodes, tensor-parallel within a node and pipeline-parallel across them. | any |
| `reproduce/repro_audit_one.sh` | Reproduces one paper and grades that exact run bundle. | any |
| `reproduce/dsv4_flash/run_dsv4_flash.sbatch` | Serves the model, then reproduces and grades a whole tier. | DeepSeek-V4-Flash, Run |
| `reproduce/dsv4_flash/retrain_dsv4_flash.sbatch` | Same, one tier up. | DeepSeek-V4-Flash, Retrain |
| `reproduce/dsv4_flash/reimplement_dsv4_flash.sbatch` | Same, one tier up. | DeepSeek-V4-Flash, Reimplement |
| `reproduce/qwen3_27b/run_qwen3_27b.sbatch` | Serves the model, then reproduces and grades a whole tier. | Qwen3.6-27B, Run |
| `reproduce/qwen3_27b/retrain_qwen3_27b.sbatch` | Same, one tier up. | Qwen3.6-27B, Retrain |
| `reproduce/qwen3_27b/reimplement_qwen3_27b.sbatch` | Same, one tier up. | Qwen3.6-27B, Reimplement |
| `reproduce/minimax_m2/run_minimax_m2.sbatch` | Serves the model, then reproduces and grades a whole tier. | MiniMax-M2.7, Run |
| `reproduce/minimax_m2/retrain_minimax_m2.sbatch` | Same, one tier up. | MiniMax-M2.7, Retrain |
| `reproduce/minimax_m2/reimplement_minimax_m2.sbatch` | Same, one tier up. | MiniMax-M2.7, Reimplement |
| `reproduce/muse_spark/muse_spark_sweep.sh` | Sweep driver for a hosted API, so it serves nothing. Run a tier entrypoint instead of this. | Muse Spark 1.2, tier from `$TIER` |
| `reproduce/muse_spark/run_muse_spark.sh` | Tier entrypoint for the driver above. | Muse Spark 1.2, Run |
| `reproduce/muse_spark/retrain_muse_spark.sh` | Tier entrypoint for the driver above. | Muse Spark 1.2, Retrain |
| `reproduce/muse_spark/reimplement_muse_spark.sh` | Tier entrypoint for the driver above. | Muse Spark 1.2, Reimplement |

Easy, Medium and Hard are the split files' stored spellings for Run, Retrain and
Reimplement. The Qwen3.6-27B scripts serve the `Qwen/Qwen3.6-27B-FP8` checkpoint.

A tier job runs its papers in ascending order of compute ceiling, grades each run while
the sweep is still going, and records a paper that produced no bundle as a harness
failure. `RESUME=1` resumes a killed job: it skips papers whose verdict is already on
disk, regrades nothing, and retries papers whose run ended without a verdict.

## Placeholders

Each placeholder is defined by a comment the first time it appears in a file.

| Placeholder | Meaning | Example shape |
| --- | --- | --- |
| `<account>` | Slurm account to charge, for this job and for the allocations the agent takes out. | `abc-project` |
| `<partition>` | Slurm partition holding the GPU nodes. | `gpu` |
| `<home>` | Home root holding the checkout, so the checkout is `<home>/reclaim`. | `/home/you` |
| `<venv>` | Virtualenv root. The scripts source `<venv>/bin/activate`. | `/home/you/reclaim/.venv` |
| `<scratch>` | Scratch or work root with room for run bundles, per-paper venvs, and compiler caches. | `/scratch/you` |
| `<hf-cache>` | Hugging Face cache directory. | `/scratch/you/hf_cache` |
| `<sif>` | Apptainer image every agent shell step runs inside, built by `cluster/build_cuda_sandbox.sh`. | `/scratch/you/cuda-agent.sif` |
| `<module>` | Module that puts Python on `PATH`. Drop the `module load` line if the virtualenv is enough. | `python/3.11` |

No script here uses a `<qos>`, a `<reservation>`, a separate `<cpu-partition>`, or a
named `<login-host>`, so those never appear.

## Assumptions

- Slurm with GPU nodes. The sweep jobs serve a model inside their own allocation, and
  the agent allocates a separate GPU node for each experiment step it runs.
- Apptainer on the compute nodes, with the image from `cluster/build_cuda_sandbox.sh`.
- The served model is reachable from every compute node the agent lands on. The serving
  layer publishes its address to a file on a shared filesystem, and consumers read it
  through `RECLAIM_ENDPOINT_FILE`.
- The harness is invoked as `PYTHONPATH=src python3 -m reclaim_repro` for a run and
  `PYTHONPATH=src python3 -m reclaim_serve` for the server. The auditor is
  `src/run_arxiv_prompt_vllm.py --mode audit`.
- The frozen split files ship in `data/splits`. The scripts point both the agent and the
  auditor at the same file, so both read the same claim for a paper.
- Telemetry upload is optional. It is off unless `RECLAIM_TELEMETRY_URL` and
  `RECLAIM_TELEMETRY_KEY` are both exported, and everything still lands on disk without
  them.
