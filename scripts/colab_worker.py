"""Autonomous Colab GPU worker for the fair-FL R&D loop.

Runs FOREVER in a Colab GPU session:
  loop:
    git pull --rebase        (pick up Claude's latest code + new jobs)
    for each job in experiments/queue.json with no results/auto/<id>/summary.csv:
        run the (algorithm x dataset x seed) grid on the GPU
        write results/auto/<id>/summary.csv (+ summary_agg.csv)
        git add / commit / push        (results flow back to the repo for Claude to read)
    sleep, repeat

The GitHub token is read from the env var GH_TOKEN (set by the notebook from a Colab Secret)
or, failing that, from google.colab.userdata.get('GH_TOKEN'). The token value is NEVER printed.
Without a token the worker still runs and commits locally but cannot push.

Launch from notebooks/colab_worker.ipynb (which installs deps + sets GH_TOKEN first).
"""
import glob
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)  # repo root = parent of scripts/
REPO = os.environ.get("MANGO_REPO") or (
    _REPO_ROOT if os.path.isdir(os.path.join(_REPO_ROOT, ".git")) else "/content/mango"
)
POLL_SECONDS = int(os.environ.get("WORKER_POLL_SECONDS", "60"))
# auto -> CUDA (Colab) > MPS (Apple Silicon) > CPU, resolved by run_experiments._select_device.
WORKER_DEVICE = os.environ.get("WORKER_DEVICE", "auto")
REMOTE_SLUG = os.environ.get("MANGO_REMOTE_SLUG", "muzakkirhussain011/mango")


def git(*args, check=False):
    r = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True)
    if r.returncode != 0 and args[:1] != ("diff",):
        print("[git]", " ".join(args), "->", ((r.stderr or r.stdout).strip()[:300]))
    if check and r.returncode != 0:
        raise RuntimeError("git %s failed" % (args,))
    return r


def _get_token():
    token = os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    try:
        from google.colab import userdata  # only available inside the Colab kernel
        return (userdata.get("GH_TOKEN") or "").strip() or None
    except Exception as e:  # noqa: BLE001
        print("[worker] GH_TOKEN not found in env or Colab secrets:", e)
        return None


def setup_remote():
    git("config", "user.email", "colab-worker@users.noreply.github.com")
    git("config", "user.name", "colab-gpu-worker")
    token = _get_token()
    if token:
        git("remote", "set-url", "origin",
            "https://x-access-token:%s@github.com/%s.git" % (token, REMOTE_SLUG))
        print("[worker] remote configured with token -> results WILL be pushed")
        return True
    print("[worker] NO TOKEN -> results committed locally only (not pushed). "
          "Add a Colab Secret named GH_TOKEN to enable push.")
    return False


def run_experiment(algo, dataset, sattr, seed, job, save_root):
    save_dir = "%s/%s/%s/seed%d" % (save_root, dataset, algo, seed)
    cmd = [sys.executable, "-m", "faircare.experiments.run_experiments",
           "--dataset", dataset, "--algorithm", algo, "--sensitive_attr", sattr,
           "--rounds", str(job.get("rounds", 40)),
           "--local_epochs", str(job.get("local_epochs", 2)),
           "--num_clients", str(job.get("num_clients", 20)),
           "--dirichlet_alpha", str(job.get("dirichlet_alpha", 0.3)),
           "--seed", str(seed), "--device", WORKER_DEVICE, "--save_dir", save_dir]
    print("[run]", algo, dataset, sattr, "seed", seed, flush=True)
    r = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    row = {"algorithm": algo, "dataset": dataset, "sensitive": sattr, "seed": seed}
    if r.returncode != 0:
        print("  FAILED:", (r.stderr or r.stdout).strip()[-800:], flush=True)
        row["error"] = (r.stderr or r.stdout).strip()[-300:]
        return row
    files = sorted(
        glob.glob(os.path.join(REPO, save_dir, "**", "final_results.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not files:
        row["error"] = "no final_results.json"
        return row
    with open(files[-1]) as fh:
        m = json.load(fh)["final_metrics"]
    row.update({
        "accuracy": m.get("test/accuracy"), "auroc": m.get("test/auroc"),
        "worst_group_f1": m.get("test/worst_group_f1"), "macro_f1": m.get("test/macro_f1"),
        "eo_gap": m.get("test/eo_gap"), "fpr_gap": m.get("test/fpr_gap"),
        "sp_gap": m.get("test/sp_gap"),
    })
    return row


def _write_csv(rows, path):
    import csv
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def do_job(job):
    jid = job["id"]
    save_root = "results/auto/%s" % jid
    out_csv = os.path.join(REPO, save_root, "summary.csv")
    if os.path.exists(out_csv):
        return False  # already completed
    print("[worker] === starting job:", jid, "===", flush=True)
    rows = []
    for ds, sattr in job["datasets"]:
        for algo in job["algorithms"]:
            for seed in job["seeds"]:
                rows.append(run_experiment(algo, ds, sattr, seed, job, save_root))
    _write_csv(rows, out_csv)
    # aggregate mean over seeds for a readable leaderboard
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        num = df.select_dtypes("number").columns
        agg = df.groupby(["dataset", "algorithm"])[list(num)].mean().round(4)
        agg.to_csv(os.path.join(REPO, save_root, "summary_agg.csv"))
        print("[worker] leaderboard for", jid, "\n", agg.to_string(), flush=True)
    except Exception as e:  # noqa: BLE001
        print("[worker] agg failed:", e)
    print("[worker] === job done:", jid, "->", out_csv, "===", flush=True)
    return True


def main():
    print("[worker] starting. repo=%s poll=%ss" % (REPO, POLL_SECONDS), flush=True)
    can_push = setup_remote()
    while True:
        git("pull", "--rebase", "--autostash")
        try:
            with open(os.path.join(REPO, "experiments", "queue.json")) as fh:
                queue = json.load(fh)
        except Exception as e:  # noqa: BLE001
            print("[worker] cannot read queue.json:", e)
            queue = {"jobs": []}
        did_any = False
        for job in queue.get("jobs", []):
            try:
                if do_job(job):
                    did_any = True
                    git("add", "results/auto")
                    git("commit", "-m", "auto: results for job %s" % job["id"])
                    git("pull", "--rebase", "--autostash")
                    if can_push:
                        git("push", "origin", "main")
            except Exception as e:  # noqa: BLE001
                print("[worker] job", job.get("id"), "crashed:", e, flush=True)
        if not did_any:
            print("[worker] no pending jobs; sleeping %ss (edit queue.json to add work)" % POLL_SECONDS,
                  flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
