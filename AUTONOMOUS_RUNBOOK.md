# AUTONOMOUS RUNBOOK — Resume the FedGMA R&D loop on a MacBook (M5)

> **Purpose.** Everything needed to resume this research on a fresh Mac and run it **autonomously**,
> with **Claude usage kept low (target ≤25% of your weekly quota)**. Written so that a new Claude Code
> session can read this file + `RESEARCH_LOG.md` and continue with no other context.
>
> Companion docs: `HANDOFF.md` (code state + bug ledger), `RESEARCH_LOG.md` (living leaderboard +
> iteration log), `research/FEDGMA_DESIGN.md` (the method + full experiment plan), `research/ADVERSARIAL_CRITIQUE.md`.

---

## 0. 60-second start (once the one-time setup in §3 is done)

Open **two terminals** on the Mac, both in the repo:

```bash
# ── Terminal 1: the WORKER (free compute — runs experiments, pushes results) ──
cd ~/mango && git pull
export GH_TOKEN=YOUR_PAT            # or: gh auth login (either works)
bash scripts/run_worker.sh          # loops forever; leave it running

# ── Terminal 2: the DRIVER (Claude Code — analyzes results, improves FedGMA) ──
cd ~/mango
claude --dangerously-skip-permissions      # start Claude Code in no-prompt (autonomous) mode
# then paste the RESUME PROMPT from §5 into Claude
```

That's the whole system. The worker grinds through the queued experiments (costs **zero** Claude
usage); Claude wakes occasionally to analyze a finished batch and improve the method.

---

## 1. The mission (and the honest finish line)

**Goal.** Build a state-of-the-art fair federated-learning method for **bias detection + mitigation**
that beats the field, at NeurIPS rigor. The method is **FedGMA** (already implemented): a *no-regret
mixture-of-aggregators* — a bank of correctly-implemented fair-FL rules blended by Hedge over a convex
surrogate, giving `O(√(T ln M))` regret (provably no worse than the best single aggregator in hindsight).

**Honest stop condition (realistic — use this, not "win every cell").**
FedGMA is **best-or-statistically-tied-best overall** across datasets for worst-group-F1 and EO/FPR/SP
gaps, accuracy held, **without per-dataset tuning**, across **≥5 seeds** with paired Wilcoxon +
Holm–Bonferroni significance — plus the ablations in `research/FEDGMA_DESIGN.md §5`. Literal strict
domination of every baseline on every metric×dataset is **not** a realistic bar (the adversarial
critique explains why) and would make the loop run forever. Report negatives honestly; **never
fabricate a number** — every value must trace to a committed results file.

---

## 2. How the autonomous loop works (architecture)

Two independent processes that talk **only through GitHub** (`main` branch):

```
   ┌────────────────────────────┐        push results         ┌─────────────────────────────┐
   │  WORKER  (scripts/          │  ──────────────────────▶   │  DRIVER  (Claude Code loop) │
   │  colab_worker.py)           │   results/auto/<id>/*.csv   │  reads results → analyzes → │
   │  • git pull                 │                             │  updates RESEARCH_LOG →     │
   │  • run queued experiments   │  ◀──────────────────────    │  improves faircare/algos/   │
   │    on CPU/MPS (FREE compute)│   push code + new jobs      │  fedgma.py → pushes →       │
   │  • push results, repeat     │   (experiments/queue.json)  │  re-queues → sleeps         │
   └────────────────────────────┘                             └─────────────────────────────┘
```

- **Worker** = the free compute engine. Runs `experiments/queue.json` jobs, writes
  `results/auto/<job_id>/summary.csv` + `summary_agg.csv`, commits + pushes. Skips finished jobs.
- **Driver** = Claude Code. Pulls results, fills the leaderboard, improves FedGMA, appends the next job.
- They never need to run on the same machine; GitHub is the shared bus. (You can even run the worker
  on the Mac and let Claude run elsewhere — but the simplest is both on the M5.)

---

## 3. One-time setup on the M5

```bash
# 1. Tools (Homebrew, Python, GitHub CLI, Claude Code)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python git gh
npm install -g @anthropic-ai/claude-code     # or the current Claude Code install method

# 2. Clone the repo
git clone https://github.com/muzakkirhussain011/mango.git ~/mango
cd ~/mango

# 3. Python env + deps (installs PyTorch for Apple Silicon — one-time, ~1-2 min)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 4. GitHub auth (pick ONE):
gh auth login                 # easiest — configures git push for BOTH worker and Claude
#   —or— create a fine-grained PAT (repo: muzakkirhussain011/mango, Contents: Read+Write) and:
#   export GH_TOKEN=github_pat_xxx     # the worker reads this; keep it secret, never share it
```

**Autonomous / no-permission mode for Claude Code:** launch with `claude --dangerously-skip-permissions`
so the driver loop can `git pull` / edit `fedgma.py` / `git push` without a prompt each time. (If you
prefer tighter control, instead pre-authorize the specific tools in `.claude/settings.local.json` and
run normally — but skip-permissions is what "fully autonomous" needs.)

---

## 4. Run the worker (Terminal 1)

```bash
cd ~/mango && git pull
export GH_TOKEN=YOUR_PAT      # skip if you used `gh auth login`
bash scripts/run_worker.sh    # default device=cpu (rock-solid). WORKER_DEVICE=mps for the Apple GPU.
```

- It installs deps, then loops: pull → run pending jobs → push results → repeat every 60s.
- Verify it works: within ~1 min the **smoke job** (`iter1a_fedgma_smoke`) finishes and you'll see a
  commit `auto: results for job iter1a_fedgma_smoke` and `results/auto/iter1a_fedgma_smoke/summary.csv`.
- **Restart anytime** (`Ctrl-C` then re-run) — it skips already-finished jobs. No Colab, no disconnects.
- If push fails: it logs `PUSH FAILED` — fix `GH_TOKEN` / `gh auth login`; results stay committed locally.

---

## 5. Run the autonomous driver (Terminal 2 — Claude Code)

Start Claude Code (`claude --dangerously-skip-permissions`) in `~/mango` and **paste this resume prompt**:

```
Resume the autonomous FedGMA R&D loop. First read AUTONOMOUS_RUNBOOK.md, RESEARCH_LOG.md, and
research/FEDGMA_DESIGN.md. Then run this self-paced loop and keep it going:

/loop FedGMA R&D driver loop (self-paced). Each iteration: (1) git pull --rebase origin main.
(2) If a NEW COMPLETED grid exists in results/auto/*/summary_agg.csv since last check: read it;
update the leaderboard + iteration log in RESEARCH_LOG.md with REAL numbers only; then act — if
FedGMA rows carry an error, fix faircare/algos/fedgma.py and push; else compare FedGMA vs baselines
and make ONE lean improvement (expert bank / convex surrogate / duals / learned gate, per
research/FEDGMA_DESIGN.md §5,§7), push, and append the next job to experiments/queue.json.
(3) STOP when FedGMA is best-or-statistically-tied-best overall on worst-group-F1 and EO/FPR/SP
gaps with accuracy held, across >=5 seeds with paired Wilcoxon + Holm-Bonferroni — update
RESEARCH_LOG.md, tell me, and stop. (4) If no new completed grid, self-pace ~1h and do NOT tight-poll.

USAGE BUDGET (hard rules, target <=25% of weekly quota):
- NEVER run a multi-agent Workflow or fan-out subagents without asking me first (these cost ~100x a
  normal turn and are the main budget risk).
- Engage expensively ONLY when a whole grid completes; keep idle checks to a bare git pull.
- One lean improve-turn per completed grid; batch, don't iterate per-run.
- Never fabricate numbers; every value traces to a committed results file.
```

The loop then drives itself: pulls, and only when a full grid lands does it spend real tokens to
analyze + improve. Between batches it's near-idle. **Pause anytime** by telling Claude "stop the loop."

---

## 6. Usage-budget policy (≤25% weekly) — READ THIS

**Cost model:** the worker training models = **$0 Claude usage** (it's Python on your Mac). Claude usage
comes only from the driver's turns. The **single biggest risk is multi-agent Workflows** (~500k–750k
tokens each — the one-time SOTA research already used one). Rules the driver must follow:

1. **No Workflows / no fan-out agents without your explicit approval.** All iteration in lean single turns.
2. **Front-loaded queue** — big batches are pre-queued so the free worker runs for hours before Claude
   engages. (Current: 7 algos × 4 datasets × **5 seeds** = 140 runs in one batch.)
3. **Batch engagement** — Claude spends real tokens only when a full grid's `summary_agg.csv` appears;
   one analyze+improve turn per batch, then it goes quiet.
4. **Slow cadence** — idle checks are a bare `git pull` at ≥1h spacing.
5. **You control it** — watch usage in the Claude app; say "pause"; Claude checkpoints before anything
   non-trivial. (Note: this is managed to "low and bounded," not a hard numeric 25% — there is no
   in-loop meter; if usage climbs, pause and re-queue a bigger batch so the free worker does more.)

---

## 7. The method & where the code lives

- **FedGMA implementation:** `faircare/algos/fedgma.py` (registered as `fedgma`).
  Expert bank: FedAvg, real q-FFL (q∈{0.5,2}), real AFL, FairFed, group-DRO → Hedge blend over the
  convex surrogate `L_t(w)=Σ w_k·loss_k + Σ_c λ_c|Σ w_k·Δr^c_k|` + slow dual ascent on EO/FPR/SP.
- **Wired into** `faircare/experiments/run_experiments.py` (algorithm `fedgma`, clean `compute_weights`
  seam — NOT the orphaned `faircare_fl.aggregate()` path).
- **Full spec + improvement backlog:** `research/FEDGMA_DESIGN.md` §4 (method), §5 (ablations), §7
  (prioritized checklist — surrogate refinement, learned gate `g_φ`, DFBD training, distillation, etc.).
  The driver improves FedGMA by working down §7 one lean step at a time.

---

## 8. Experiments contract (how the two sides communicate)

- **Add work:** append a job to `experiments/queue.json`:
  ```json
  { "id": "iterN_shortname", "note": "...", "algorithms": ["fedavg","fedgma", ...],
    "datasets": [["adult","sex"],["compas","race"],["diabetes130","race"],["synth_health","sex"]],
    "seeds": [0,1,2,3,4], "rounds": 40, "local_epochs": 2, "num_clients": 20, "dirichlet_alpha": 0.3 }
  ```
  Give each job a **unique `id`** — the worker runs any job whose `results/auto/<id>/summary.csv` is absent.
- **Read results:** `results/auto/<id>/summary.csv` (per-run rows) and `summary_agg.csv` (mean over seeds).
  Columns: accuracy, auroc, worst_group_f1, macro_f1, eo_gap, fpr_gap, sp_gap (+ `error` if a run failed).
- **Datasets / sensitive attrs:** adult(sex/race), compas(race/sex), diabetes130(race/gender),
  synth_health(auto), heart(sex/age). mimic/eicu are synthetic stubs — not real EHR.

---

## 9. Current state snapshot (as of this handoff)

- ✅ Real-data pipeline works on GPU/MPS/CPU (verified on Colab T4: Adult AUROC 0.73 — not noise).
- ✅ New real loaders: `adult, heart, diabetes130, compas, synth_health` (diabetes130/compas download on
  first use — the smoke job validates them).
- ✅ FedGMA implemented, wired, pushed. Baselines' q-FFL/AFL fixed (were inverted).
- ✅ Autonomous worker + queue built; usage policy defined.
- ⏳ **No experiments have completed yet** — the worker has not been run on a stable machine. First job
  to run: `iter1a_fedgma_smoke` (fast), then `iter1_fedgma_vs_baselines` (140 runs). `RESEARCH_LOG.md`
  leaderboard is empty until then.
- Latest `main` commit is this handoff. Check `git log --oneline -5` for the exact HEAD.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Worker: `PUSH FAILED` | `export GH_TOKEN=...` in that shell, or `gh auth login`; then re-run. Results are safe (committed locally). |
| `python3: command not found` | `brew install python`, or `PYTHON=python bash scripts/run_worker.sh`. |
| MPS dtype error mid-run | Use CPU: `WORKER_DEVICE=cpu bash scripts/run_worker.sh` (tiny models — CPU is plenty). |
| FedGMA run has `error` in summary.csv | The driver reads the traceback and fixes `faircare/algos/fedgma.py`. If doing it by hand, run one cell manually: `python -m faircare.experiments.run_experiments --dataset adult --algorithm fedgma --sensitive_attr sex --rounds 10 --num_clients 10 --device cpu --save_dir results/debug`. |
| Baseline AUROC ≈ 0.5 (degenerate) | Data path / config regression — see `HANDOFF.md` §6 validation gate before trusting any number. |
| Claude keeps prompting for permission | Relaunch with `claude --dangerously-skip-permissions`. |
| Want to reduce Claude usage further | Queue a bigger batch (more seeds/ablations) so the free worker does more per Claude turn; widen the loop cadence; tell Claude "pause". |

---

## 11. What "done" looks like

`RESEARCH_LOG.md` leaderboard shows FedGMA at or above every baseline on the fairness metrics with
accuracy held, across ≥5 seeds, with significance — and the ablations in `FEDGMA_DESIGN.md §5` are filled
in. The driver announces the result and stops the loop. At that point the material for a NeurIPS write-up
(method, honest results table, ablations) is all in the repo.
