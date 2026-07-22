# RESEARCH LOG — Toward SOTA Fair Federated Learning (Bias Detection + Mitigation)

**Mission:** Build a state-of-the-art federated-learning method for **bias detection and mitigation**
that provably surpasses existing FL models, at **NeurIPS-level** rigor. Core approach (mandate): an
**ensemble** that adapts the best parts of the best fair-FL algorithms **plus our own novel logic**.

This is the living tracker for a continuous research→design→implement→evaluate→iterate loop. It is
updated every iteration and is the source of truth for "are we winning yet."

---

## Ground rules (non-negotiable, NeurIPS bar)
1. **Every SOTA claim is earned, never asserted.** A win counts only with real, seeded (≥5),
   multi-dataset results, statistical significance (paired tests + Holm–Bonferroni), and honest CIs.
2. **No fabricated numbers.** All numbers trace to a committed `final_results.json` / results CSV.
3. **Fair comparison.** Same data splits, seeds, rounds, model, and tuning budget across all methods.
4. **Report negatives.** If our method loses on a dataset/metric, we record it and iterate — we do not
   hide it. (The previous incarnation of this repo fabricated wins; we are the correction.)
5. **Ablations required.** Each novel component must be shown to help via an ablation.

## Operating model (the loop)
```
[Research online: SOTA + ensembles]  ->  [Design novel ensemble method]
        ->  [Implement in faircare/algos/]  ->  [GPU sweep on Colab T4 (notebooks/colab_research.ipynb)]
        ->  [Analyze vs baselines in this log]  ->  [Improve]  -> repeat until SOTA on all datasets
```
Compute: Colab **T4 GPU** (primary, via the notebook) + the user's **MacBook M4 Max (MPS)**. Claude
drives research/design/implementation and analyzes results; GPU runs are launched on Colab.

---

## Status board

| Item | State |
|---|---|
| Pipeline trains on REAL data (not noise) | ✅ verified on Colab GPU (Adult smoke: acc 0.752, AUROC 0.727) |
| Apple MPS + CUDA device support | ✅ |
| Real loaders: adult, heart, diabetes130, compas, synth_health | ✅ wired (diabetes130/compas download-validated: _pending cell 7_) |
| Honest baseline table (6 algos × 4 datasets × ≥3 seeds) | ⏳ in progress |
| SOTA literature survey (online) | ⏳ research workflow running |
| Novel ensemble method designed | ⏳ (design phase of research workflow) |
| Novel method implemented | ☐ |
| Novel method BEATS baselines (stat-sig, all datasets) | ☐ ← the finish line |

---

## Leaderboard (fill with REAL numbers as runs land)
Target metrics: **worst-group-F1 ↑**, **EO/FPR/SP gap ↓**, accuracy/AUROC held. Best per column in **bold**.

### Adult (sensitive: sex)
| Method | Acc | AUROC | worst-grp F1 | EO gap | FPR gap | SP gap |
|---|---|---|---|---|---|---|
| _baselines + ours — pending full sweep_ | | | | | | |

_(Repeat per dataset: COMPAS, Diabetes-130, synth_health.)_

---

## Iteration log

### Iter 0 — Establish a valid baseline (in progress)
- Fixed the root cause (runner trained on Gaussian noise → now real data). Verified on Colab GPU.
- Running the 30-round validation gate (all 6 baselines on Adult) + controlled-bias + new-loader checks.
- Next: full 6×4×seeds sweep → fill the leaderboard → this is the bar our method must clear.

### Iter 1 — SOTA research + novel design (in progress)
- Background workflow: online survey of fair-FL aggregators, bias detection/mitigation, ensembles/MoE/
  distillation, benchmarks, and MOO; then 3 diverse novel-ensemble designs; adversarial novelty/
  feasibility critique; synthesized spec. Deliverables land in `research/`.

### Iter 2+ — Implement → run → analyze → improve
- _to be filled_
