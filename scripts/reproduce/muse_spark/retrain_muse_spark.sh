#!/usr/bin/env bash
# Reproduce and grade every Retrain-tier paper of the frozen benchmark on the hosted model API.
# Assumes $META_MODEL_KEY and the placeholders filled in muse_spark_sweep.sh.
# Env overrides (SPLIT, MODEL, MAX_PARALLEL, ONLY_IDS, BUDGET_H100_HOURS, RESUME) are documented there.
# Run it detached: nohup scripts/reproduce/muse_spark/retrain_muse_spark.sh > retrain_muse.log 2>&1 &
set -uo pipefail
# stored spelling of the Run|Retrain|Reimplement tier in the split files
TIER=Medium exec "$(dirname "${BASH_SOURCE[0]}")/muse_spark_sweep.sh" "$@"
