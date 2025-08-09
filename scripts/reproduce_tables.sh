# scripts/reproduce_tables.sh
set -e
python -m faircare.experiments.run_sweep --dataset heart --algos faircare fedgft qffl afl --seeds 42 43 44 45 46 --outdir runs/heart_sweep/
python -m faircare.paper.tables --indir runs/heart_sweep/ --outdir paper/tables/
