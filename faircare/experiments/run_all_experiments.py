#!/usr/bin/env python3
"""
run_all_experiments.py

Runs all algorithm × dataset combinations defined in run_experiments.py with:
  --rounds 20  --local_epochs 5
Captures stdout to per-run logs and aggregates the "Final Results" summary into JSON.

Usage:
  python run_all_experiments.py
  python run_all_experiments.py --seeds 0 1 --max-workers 2 --device auto
  python run_all_experiments.py --runner-path /path/to/run_experiments.py
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from typing import Dict, List, Tuple, Optional

# Match lines like "  final_accuracy: 0.7845"
RESULT_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$")
FINAL_RESULTS_START = re.compile(r"^\s*Final Results:\s*$")
RESULTS_SAVED_RE = re.compile(r"^\s*Results saved to:\s*(.+)\s*$")

# Keep these in sync with run_experiments.py choices
ALGOS = ["fedavg", "fedprox", "qffl", "afl", "fairfate", "faircare_fl", "fairfed"]
DATASETS = ["adult", "heart", "synth_health", "mimic", "eicu"]

def guess_device(auto_choice: str) -> str:
    if auto_choice != "auto":
        return auto_choice
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def build_cmd(
    runner_path: Path,
    algo: str,
    dataset: str,
    seed: int,
    device: str,
    rounds: int,
    local_epochs: int,
    extra: List[str],
) -> List[str]:
    cmd = [
        sys.executable,
        str(runner_path),
        "--algo", algo,
        "--dataset", dataset,
        "--rounds", str(rounds),
        "--local_epochs", str(local_epochs),
        "--seed", str(seed),
        "--device", device,
    ]
    if extra:
        cmd.extend(extra)
    return cmd

def run_one(
    runner_path: Path,
    algo: str,
    dataset: str,
    seed: int,
    device: str,
    rounds: int,
    local_epochs: int,
    logs_dir: Path,
    extra_args: List[str],
    env: Dict[str, str],
) -> Tuple[str, str, int, Dict[str, float], Optional[str], str]:
    """
    Returns:
      (algo, dataset, seed, metrics_dict, results_logdir, log_path)
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{algo}_{dataset}_seed{seed}.log"

    cmd = build_cmd(runner_path, algo, dataset, seed, device, rounds, local_epochs, extra_args)

    metrics: Dict[str, float] = {}
    results_logdir: Optional[str] = None
    final_block = False

    with Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, env=env) as proc, log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"# CMD: {' '.join(cmd)}\n")
        lf.write(f"# START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for line in proc.stdout:  # type: ignore
            lf.write(line)
            # Parse "Final Results" section
            if FINAL_RESULTS_START.match(line):
                final_block = True
                continue
            if final_block:
                m = RESULT_LINE_RE.match(line)
                if m:
                    k, v = m.group(1), float(m.group(2))
                    metrics[k] = v
                # Stop parsing the block on empty line or "Results saved to:"
                if not line.strip():
                    final_block = False
                    continue
            # Also try to capture the "Results saved to:" path
            m2 = RESULTS_SAVED_RE.match(line)
            if m2:
                results_logdir = m2.group(1).strip()

        rc = proc.wait()

    return (algo, dataset, seed, rc, metrics, results_logdir, str(log_path))

def main():
    parser = argparse.ArgumentParser(description="Sweep all algos × datasets for run_experiments.py")
    parser.add_argument("--runner-path", type=str, default="run_experiments.py",
                        help="Path to run_experiments.py (default: ./run_experiments.py)")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0], help="Seeds to run (default: 0)")
    parser.add_argument("--rounds", type=int, default=20, help="Training rounds (forced default: 20)")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs (forced default: 5)")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto (default: auto)")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel runs (default: 1)")
    parser.add_argument("--logs-dir", type=str, default="sweep_logs", help="Directory for stdout logs")
    parser.add_argument("--extra-args", type=str, nargs=argparse.REMAINDER,
                        help="Any extra args to pass through to run_experiments.py (everything after --)")
    args = parser.parse_args()

    runner_path = Path(args.runner_path).resolve()
    if not runner_path.exists():
        print(f"ERROR: run_experiments.py not found at {runner_path}", file=sys.stderr)
        sys.exit(1)

    device = guess_device(args.device)
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    aggregated_path = logs_dir / "aggregated_results.json"

    # Ensure reproducible threading in BLAS libs for fairness across runs
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    combos = [(a, d, s) for a in ALGOS for d in DATASETS for s in args.seeds]

    print(f"Runner: {runner_path}")
    print(f"Device: {device}")
    print(f"Combos: {len(combos)} ({len(ALGOS)} algos × {len(DATASETS)} datasets × {len(args.seeds)} seeds)")
    print(f"Logs dir: {logs_dir}")
    print(f"Rounds: {args.rounds} | Local epochs: {args.local_epochs}")
    if args.extra_args:
        print(f"Extra args → {' '.join(args.extra_args)}")

    results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = []
        for algo, dataset, seed in combos:
            futures.append(
                ex.submit(
                    run_one,
                    runner_path,
                    algo,
                    dataset,
                    seed,
                    device,
                    args.rounds,
                    args.local_epochs,
                    logs_dir,
                    args.extra_args or [],
                    env,
                )
            )

        for fut in as_completed(futures):
            algo, dataset, seed, rc, metrics, saved_dir, log_path = fut.result()
            row = {
                "algo": algo,
                "dataset": dataset,
                "seed": seed,
                "return_code": rc,
                "results_logdir": saved_dir,
                "log_path": log_path,
                "final_metrics": metrics,
            }
            results.append(row)
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            print(f"[{status}] {algo} × {dataset} × seed{seed}  -> log: {log_path}")

            if rc != 0:
                failures.append(row)

    # Write aggregated JSON
    with aggregated_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": device,
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "seeds": args.seeds,
                "algos": ALGOS,
                "datasets": DATASETS,
                "results": results,
                "failures": failures,
            },
            f,
            indent=2,
        )

    print(f"\nAggregated results written to: {aggregated_path}")
    if failures:
        print("Some runs failed. See their log files for details:")
        for r in failures:
            print(f"  - {r['algo']} × {r['dataset']} × seed{r['seed']} :: {r['log_path']}")

if __name__ == "__main__":
    main()
