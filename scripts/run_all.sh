# scripts/run_all.sh
set -e
python -m faircare.experiments.run_experiments --dataset heart --algo faircare --num_clients 5 --rounds 20 --local_epochs 1 --batch_size 64 --lr 1e-3 --lambdaG 2.0 --lambdaC 0.5 --lambdaA 0.5 --q 0.5 --beta 0.9 --dirichlet_alpha 0.5 --sensitive_attr sex --outdir runs/heart_faircare/
python -m faircare.paper.make_figures --indir runs/heart_faircare/ --outdir paper/figs/
python -m faircare.paper.tables --indir runs/heart_faircare/ --outdir paper/tables/
