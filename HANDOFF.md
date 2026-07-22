# MANGO / FairCare-FL — Research Handoff

> **Read this first.** It is written for a fresh Claude Code session (or human) picking up this
> research on a **MacBook Pro M4 Max**. It records what was broken, what was just fixed, how to run
> things, and the exact next steps. Last updated: 2026-07-03.

---

## 1. TL;DR — the situation

MANGO benchmarks a novel fair-federated-learning aggregator, **FairCare-FL**, against baselines
(fedavg, fedprox, qffl, afl, fairfed) for "fairness in healthcare ML."

**The previously committed results were meaningless.** The experiment runner imported its dataset
loader from a synthetic stub (`faircare/data/datasets.py::load_dataset`) that returns
`np.random.randn(...)` for *every* dataset name. So every number in
`results/full_evaluation/comprehensive_analysis.csv` and `paper/RESULTS.md` was computed on Gaussian
noise (AUROC ≈ 0.50 everywhere; FairCare-FL collapsed to a constant predictor → accuracy std 0.0, all
fairness gaps exactly 0.0). **Those artifacts must be regenerated from scratch and not trusted.**

**This session rewired the runner onto the real dataset loaders and fixed the correctness bugs that
block trustworthy numbers.** The repo is now ready to produce an honest baseline. Nothing has been
re-run yet — that is your first job (see §5, §6).

---

## 2. Bug ledger (file:line, status)

| # | Bug | Location | Status |
|---|-----|----------|--------|
| 1 | Runner trained on **random noise** (synthetic stub instead of real loaders) | `faircare/experiments/run_experiments.py:32` + `prepare_data` | **FIXED** — now imports `from faircare.data import load_dataset` and consumes the Dict contract |
| 2 | Heart **sensitive attribute misaligned** (sliced positionally from unshuffled array after shuffle) | `faircare/data/heart.py` | **FIXED** — `a` now carried through `train_test_split` |
| 3 | **Scaler leakage** (StandardScaler fit on full data before split) | `adult.py`, `heart.py` (+ new loaders) | **FIXED** — scaler fit on train only, then transform val/test |
| 4 | Adversarial **double-negation** (GRL negates AND loss term negated → encoder reveals sensitive attr) | `faircare/core/client.py` adversarial term | **FIXED** — coefficient made positive (GRL supplies the single sign flip) |
| 5 | **No Apple MPS support** (hard-coded `cuda`-or-`cpu` → silent CPU on M-series) | `run_experiments.py:48`, `client.py`, `faircare_fl.py` | **FIXED** — `_select_device()` picks CUDA > MPS > CPU; new `--device` flag |
| 6 | Config-shadow: `faircare/config/` package shadows `faircare/config.py`, so `from faircare.config import ExperimentConfig` fails | `trainer.py`, `demo_faircare_fl.py`, `tests/test_e2e_smoke.py` | **OPEN (P2)** — does **not** affect the sweep path (`run_experiments.py`/`run_sweep.py` never import `ExperimentConfig`). Fix by re-exporting/relocating `ExperimentConfig`. |
| 7 | `run_all_experiments.py` targets a deleted CLI (`--algo`/`--device`, `fairfate`) | `faircare/experiments/run_all_experiments.py` | **OPEN (P2)** — do not use it; use `run_experiments.py` (single) / `run_sweep.py` (sweep) |
| 8 | DFBD "bias detector" network is **never trained** (random tilts) | `faircare/algos/faircare_fl.py` (`dfbd_*`) | **OPEN** — cosmetic for baseline; revisit only if it matters for a claim |
| 9 | `secure_agg.py` provides **no real privacy** (server-known seeds; masks don't cancel under weighted sum) | `faircare/core/secure_agg.py` | **OPEN** — it's a research stub; SA/DP are disabled in all runs anyway |
| 10 | Differentiable fairness loss `softmax`es a single logit | `client.py::_compute_local_fairness_loss` | **NOT A BUG in practice** — models use `output_dim=2` (two logits), so softmax is valid. Only a hazard if a 1-logit head is introduced. |

`mimic`/`eicu` remain **synthetic stubs** (`faircare/data/mimic_eicu.py`) — they fall back to
`synth_health` and are **not** real ICU data (credentialed PhysioNet access required). The dispatcher
docstring now says so explicitly. Do not present mimic/eicu results as real EHR.

---

## 3. Data pipeline (post-fix)

**Loader contract (the "Dict contract").** `faircare.data.load_dataset(name, sensitive_attribute, seed)`
returns:
```python
{"train": Dataset, "val": Dataset, "test": Dataset,
 "n_features": int, "n_classes": 2, "sensitive_attribute": str}
```
where each split's `__getitem__` yields `(X: FloatTensor, y: LongTensor, a: LongTensor)`.
`faircare/experiments/run_experiments.py::prepare_data` consumes this Dict, then partitions the train
split into clients via `faircare.data.datasets.create_federated_splits(dataset, num_clients, alpha, seed)`
(Dirichlet over labels). The partitioner re-wraps clients into `FairDataset`, so all downstream client
training/eval code is unchanged — only the *source data* changed from noise to real.

**Datasets available** (downloaded + cached under `~/.faircare/data/`):

| name (`--dataset`) | Real? | `--sensitive_attr` | ~Size | Task | Notes |
|---|---|---|---|---|---|
| `adult` | ✅ real (UCI) | `sex`, `race` | 48.8k | income >50K | correctness anchor; wired + working |
| `heart` | ✅ real (UCI Cleveland) | `sex`, `age` | ~300 | heart disease | small; sanity only; bug #2 fixed |
| `diabetes130` | ✅ real (UCI id-296) | `race`, `gender` | ~101k | 30-day readmission | **best healthcare set; NEW loader — validate download on first run** |
| `compas` | ✅ real (ProPublica) | `race`, `sex` | ~7k (after filter) | 2-yr recidivism | canonical fairness set; **NEW loader — validate download** |
| `synth_health` | ⚙️ synthetic | (auto `synthetic_group`) | configurable | biased binary | controlled probe with KNOWN injected bias |
| `mimic`, `eicu` | ❌ synthetic stub | n/a | — | — | fall back to `synth_health`; not real ICU data |

**Licensing/credibility:** Adult, Heart, Diabetes-130 = UCI (open, citable). COMPAS = ProPublica
(open, widely cited). All are standard, published fairness/clinical benchmarks — credible for a paper.
`diabetes130` and `compas` loaders are **new and untested against the live network** — the very first
run downloads & caches them; confirm the download succeeds and the shapes look sane.

---

## 4. Environment setup (macOS / Apple Silicon)

```bash
cd /path/to/mango
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the package + deps from pyproject.toml
# If lint/test tooling is needed (CI installs only runtime deps — see note below):
pip install pytest ruff black
```
- PyTorch on Apple Silicon ships MPS support out of the box (`torch>=2.0`). Verify:
  `python -c "import torch; print(torch.backends.mps.is_available())"` → should be `True`.
- **MPS caveat:** MPS does not support float64. Our tensors are float32, but if you hit a
  `Cannot convert ... float64` error mid-run, just use `--device cpu` (the MLP-on-tabular is tiny, so
  CPU on an M4 Max is still fast). Recommended: run the **validation gate on `--device cpu`** first
  for reproducibility, then use `--device mps` for the full sweep speedup.
- CI note (`.github/workflows/ci.yml`): it installs only `requirements.txt` (no `ruff`/`black`/
  `pytest`), so the lint/test steps are currently red. Not blocking for research; fix if you care about CI.

---

## 5. How to run

**Fastest path — Google Colab GPU (recommended).** Open `notebooks/colab_research.ipynb` in Colab
directly from GitHub:
`https://colab.research.google.com/github/muzakkirhussain011/mango/blob/main/notebooks/colab_research.ipynb`
Then `Runtime > Change runtime type > GPU` and `Run all`. It clones `main`, installs deps, checks the
GPU, and runs the smoke test + validation gate + controlled-bias + new-loader checks. Flip
`RUN_FULL_SWEEP = True` (cell 8) for the headline grid. Colab GPUs are CUDA, which `_select_device()`
picks automatically. (`--device mps` is for the MacBook; Colab uses `--device cuda`.)

**Local — smoke test (fast, proves the pipeline end-to-end):**
```bash
python -m faircare.experiments.run_experiments \
  --dataset adult --algorithm fedavg --sensitive_attr sex \
  --num_clients 10 --rounds 5 --local_epochs 1 --device cpu \
  --save_dir results/smoke
```
Expect: it downloads Adult once (cached), logs `Using device: cpu`, completes without error, and
writes `results/smoke/.../final_results.json`.

**Single real run:**
```bash
python -m faircare.experiments.run_experiments \
  --dataset diabetes130 --algorithm faircare_fl --sensitive_attr race \
  --num_clients 20 --rounds 50 --local_epochs 2 --dirichlet_alpha 0.3 \
  --device mps --save_dir results/full_evaluation
```

**Full sweep** (algorithm × dataset × seed): use `faircare/experiments/run_sweep.py`
(`ComprehensiveSweep`) — it runs `FederatedExperiment` in-process with a `ProcessPoolExecutor`.
Start small (2 algos × 1 dataset × 1 seed, `rounds=5`) to confirm plumbing, then scale to the headline
grid. MPS is a single device, so don't assume per-worker GPU — cap workers to the performance-core count.

---

## 6. Validation gate — DO THIS BEFORE TRUSTING ANY NUMBER

All must hold, or the data path / a bug is still wrong:
1. **Adult FedAvg test AUROC ≈ 0.85–0.90** (NOT ~0.51). If ~0.5, the loader isn't returning real data.
2. **Accuracy std across seeds > 0** (the old degenerate runs had std 0.0).
3. **Fairness gaps non-zero and varying by algorithm** (old runs had all gaps exactly 0.0).
4. **Controlled-bias check:** on `synth_health` (known `bias_level`), FairCare-FL should **reduce** the
   EO gap vs FedAvg — proof the aggregator actually does something.

Only after the gate passes: regenerate `results/full_evaluation/comprehensive_analysis.csv` from the
fixed sweep, and regenerate `paper/RESULTS.md` **from that CSV** (never hand-edit numbers). Quarantine
the old fabricated numbers with a note.

---

## 7. Research roadmap (what's left)

**Phase 1 — Literature/SOTA survey (not started).** Use the `deep-research` skill. Produce
`research/LITERATURE_SURVEY.md`, `research/SOTA_BASELINE_TABLE.md`, `research/ENSEMBLE_DIRECTIONS.md`.
Cover fair-FL aggregators (q-FFL, AFL, FairFed, Ditto, FedMinMax), group-fairness in-processing
(Hardt 2016; Agarwal 2018 reductions = `fairlearn`, already a dep; Zhang 2018 adversarial; IRM), the
MOO methods the repo already uses (MGDA/Sener 2018, PCGrad/Yu 2020, CAGrad/Liu 2021), and ensembles in
FL (FedDF/FedBE distillation, MoE gating, AdaFair boosting).

**Phase 5 — Ensemble improvement (not started; only after §6 gate passes).** Ranked, grounded in
existing (currently dead) scaffolding in `faircare/algos/faircare_fl.py`:
- **Idea A (do first — best payoff/effort): gated ensemble-of-aggregators (MoE).**
  `_compute_component_weights` (~line 1351) already builds FedAvg/FairFed/q-FFL/FedProx/AFL weight
  vectors, and a `gate_network` stub (~line 1235) exists but is never used. Wire round-level signals
  (EO/FPR/SP gaps, worst-group-F1, round idx) → gate → convex blend of the component weight vectors.
  Provably ≥ the best single component if the gate learns to select it.
- **Idea B: real server-side distillation (FedDF/FedBE).** `_perform_distillation` (~line 1085)
  returns zeros (placeholder). Distill top-K client soft predictions on the server val split.
- **Idea C (stretch): fairness-boosting client reweighting (AdaFair-style)** using the existing
  Lyapunov `fairness_debt_scores` (~line 1095), extending it from *selection* to *aggregation weight*.

Report every improvement as a **delta over the honest §6 baseline** with identical seeds/datasets and
**paired** statistical tests (paired t / Wilcoxon), Holm–Bonferroni corrected across metric×baseline.

---

## 8. Open questions / risks
- `diabetes130` & `compas` loaders are new and unvalidated against the live network — confirm on first run.
- MPS float64 fallback (see §4) — use `--device cpu` if you hit dtype errors.
- Config-shadow (bug #6) blocks `trainer.py`/`demo`/`test_e2e_smoke.py`; fix before relying on those.
- `partition.py::make_federated_splits` (richer: adds a pooled server-val) is unused by the sweep;
  consolidating the two partitioners is an optional later refactor.
- Whether to keep `mimic`/`eicu` at all, or rename them to make the synthetic nature obvious.

---

## 9. File map (what changed this session)
- `faircare/experiments/run_experiments.py` — real data path, `_select_device`, `--device`, dataset choices.
- `faircare/data/__init__.py` — registered `diabetes130` + `compas`; honest docstring.
- `faircare/data/heart.py` — sensitive-attr alignment + scaler-on-train.
- `faircare/data/adult.py` — scaler-on-train.
- `faircare/data/diabetes.py` — **new** UCI Diabetes-130 loader.
- `faircare/data/compas.py` — **new** ProPublica COMPAS loader.
- `faircare/core/client.py` — `_select_device`; adversarial sign fix.
- `faircare/algos/faircare_fl.py` — `_select_device`.
- Plan: `.claude/plans/we-need-to-continue-deep-rabbit.md` (full phased plan).
