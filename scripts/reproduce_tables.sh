#!/usr/bin/env bash
set -e

# Example: run multiple seeds and write compact JSON to paper/tables/
# Assumes configs exist for each algorithm in experiments/configs/algos.yaml
ALGS=("fedavg" "qffl" "fairfed" "fairfate" "fedgft" "faircare_f1")
DATASET="adult"
OUTDIR="paper/tables"
mkdir -p $OUTDIR

for ALG in "${ALGS[@]}"; do
  JSONOUT="${OUTDIR}/${ALG}_${DATASET}.json"
  echo "[]" > $JSONOUT
  for SEED in 1 2 3 4 5; do
    python -m faircare.experiments.run_experiments --algo $ALG --dataset $DATASET | \
      python - <<'PY'
import sys,json,re
lines=sys.stdin.read().strip().splitlines()
last=[l for l in lines if l.startswith("[Round")]
if not last: sys.exit(0)
m=eval(last[-1].split("]")[1].strip())
print(json.dumps(m))
PY
    # append to JSON array
    LAST=$(tail -n 1 nohup.out 2>/dev/null || true)
    # For simplicity, we read from stdout captured above:
    # In Colab use cell magic to capture; locally you might redirect.
  done
done

python -m paper.tables
