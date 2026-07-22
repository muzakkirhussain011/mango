All load-bearing repo facts are confirmed: dead `_compute_component_weights` (:1372, no call site), mislabeled experts (`qffl`=1/(loss²+ε) at :1402, `afl`=1/(loss+ε) at :1409, `fedprox`=clone at :1406), dead `gate_network` (:1257), stub `_perform_distillation` (:1106), untrained `dfbd_optimizer` (:186, no `.step()`), the weight-space server seam (`server.py:181`), and the degenerate results (faircare_fl gaps==0.0/std==0.0 on adult/mimic/eicu; datasets present are adult/compas/mimic/eicu, not Diabetes-130). Writing the deliverable now.

# Fair Federated Learning: Research Synthesis and Method Design for SOTA Bias Detection + Mitigation

**Deliverable version:** definitive design spec, NeurIPS-track target
**Scope:** aggregation-level fairness in FL; group + client fairness; demographics-free detection
**Chosen method:** **FedGMA — Federated Gated Mixture-of-Aggregators with No-Regret Fair Blending**
**Repository:** `D:\mango_git\mango` (`faircare/`)

---

## 1. Executive summary

**The problem.** Every deployed fair-FL aggregator fixes **one** point on the accuracy–fairness frontier through **one** knob (q-FFL's `q`, FairFed's `β`, Ditto/GIFAIR's `λ`, AdaFed's `γ`, PropFair's `M`, FedMinMax's simplex floor `ε`) that must be re-tuned per dataset. Worse, two fairness families — **client/performance fairness** (uniform accuracy across devices) and **demographic-group fairness** (SPD/EOD/DP/EO across sensitive groups) — provably do **not** co-optimize: FedMinMax establishes that the worst client ≠ the worst group in general [Papadaki et al., FAccT 2022]. No single monolithic model dominates on both axes at once.

**The chosen contribution.** We cast **fair-FL aggregation-rule selection as full-information online convex optimization (OCO)**. Each round the server holds a bank of M aggregation rules (FedAvg, real q-FFL, real AFL/minimax, FairFed, group-DRO), each of which maps client reports to a per-client simplex weight vector `w^(m) ∈ Δ_{K−1}`. A server-side **counterfactual surrogate** `L_t(w)` — **convex in `w` and evaluable for every rule without deploying any of them** — turns rule selection into Hedge / exponentiated-weights over the M experts. Because a convex combination of simplex vectors is itself a simplex vector, the blended weight vector `w_t = Σ_m α_{t,m} w^(m)_t` is a valid aggregator by construction, and by Jensen + Hedge it is **asymptotically no worse than the best single aggregation rule chosen in hindsight**, with regret `O(√(T ln M))`. A slow-timescale dual ascent on group-fairness constraints couples in as a primal–dual saddle-point scheme giving `gap_c ≤ ε_c + O(1/√T)`.

**Why this and not the two ensemble-distillation alternatives.** The other two designs reviewed (FairBoost-DF, BCPE-FL) both require infrastructure the repo does not have — a server-held probe set, a working distillation loop, materialized candidate models — and both rest their central "worst-group error ↓ via ensemble cancellation" claim on **members that are reweightings of the *same* client deltas along a *single* global trajectory**, i.e., highly correlated teachers that do not deliver independent-error cancellation. FedGMA lives **entirely in weight-space**, which is exactly the existing `compute_weights` → weighted-average interface (`faircare/core/server.py:181`), needs **no probe set and no distillation**, has the only **genuinely novel + provable** contribution of the three, and makes an **honest, defensible empirical claim** (self-tuning to the best baseline per dataset, with a regret certificate) that survives the correlated-members objection.

**The blunt precondition.** The current repo produces a **degenerate constant-class predictor** for `faircare_fl` (all fairness gaps exactly `0.0` with `std 0.0`, macro-F1 ≈ 0.31, AUROC ≈ 0.51 — `results/full_evaluation/comprehensive_analysis.csv`), and *every* baseline lands near AUROC 0.50, i.e., there is currently **no measurable worst-group-F1 and no accuracy signal to be fair about.** Additionally the "expert bank" the design depends on (`_compute_component_weights`, `faircare_fl.py:1372`) is **dead code with no call site**, and its members are **mislabeled inversions** of the baselines they claim to be (`qffl` = `1/(loss²+ε)` and `afl` = `1/(loss+ε)` both *down-weight* high-loss clients — the exact opposite of real q-FFL/AFL). **No number is reportable until (i) the pipeline is de-degenerated on real data and (ii) the baselines are implemented faithfully.** These are week-1, method-independent work items, itemized in §7.

---

## 2. SOTA landscape

### 2.1 Two fairness families (and why they don't merge)

| Family | Definition | Metric | Representative methods |
|---|---|---|---|
| **Client / performance fairness** | uniform accuracy across clients/devices | std/variance of per-device accuracy; worst-10% accuracy | q-FFL, AFL, Ditto, PropFair, GIFAIR-FL, AdaFed |
| **Demographic-group fairness** | parity across sensitive groups | SPD, EOD, DP, EO gaps; worst-group risk/F1 | FairFed, FedFB, FedMinMax, FAIR-FATE, GLocalFair, LoGoFair, FedFACT |

FedMinMax proves worst-client ≠ worst-group in general [Papadaki et al., FAccT 2022], so optimizing one does **not** deliver the other — the core motivation for carrying both as experts.

### 2.2 The reweighting primitives (the whole field reduces to seven)

1. **loss-exponent up-weighting** — q-FFL `loss^q`, TERM, AdaFed `|f|^γ`
2. **minimax over clients** (AFL) or **over groups** (FedMinMax) via ascent on simplex weights
3. **deviation-based aggregation-weight nudging** — FairFed `w_k −= β(Δ_k − mean Δ)`
4. **fairness-gated momentum averaging** — FAIR-FATE
5. **loss-spread / Nash-product regularizers** — GIFAIR-FL `|L_i−L_j|`, PropFair `log(M−F)`
6. **bilevel group reweighting** — FedFB / FairBatch
7. **post-hoc calibration** — FedFACT, LoGoFair

FedGMA's expert bank instantiates primitives 1–3; the surrogate + duals reproduce the constrained-fairness intent of primitives 2, 5, 6 online; post-hoc calibration (7) is available as an *optional* deploy-time layer but is **out of the spine**.

### 2.3 Comparison table (aggregators, detection/mitigation, ensembles, MOO)

| Method | Family / role | Mechanism primitive | Fairness notion | Single knob | Datasets | Headline result (provenance) | Venue | Key weakness (attack surface) |
|---|---|---|---|---|---|---|---|---|
| **FedAvg** | baseline | sample-prop average | none | — | all | reference | McMahan 2017 | no fairness |
| **q-FFL / q-FedAvg** | client-fair | loss^q up-weight | worst-client | q | Vehicle, Synthetic, Sent140, Shakespeare | Vehicle worst-10% 43.0→69.9%, var 291→48 at ~87% mean (**verified, Table 1**) | ICLR 2020 | one op-point; no group notion |
| **AFL** | client-fair | minimax over clients | worst-client | — | Adult, Fashion-MNIST | theoretical minimax bound; small worst-client gains | ICML 2019 | avg-accuracy collapse; degrades w/ #clients |
| **Ditto** | personalization | `F_k(v)+λ/2‖v−w*‖²` | per-device std | λ | Fashion-MNIST, FEMNIST | FMNIST 0.930 vs TERM 0.637 under 50% poison (**reported**) | ICML 2021 | client-fair only; no DP/EO |
| **PropFair** | client-fair | Nash `log(M−F)` | proportional | M | CIFAR-10/100, TinyImageNet | CIFAR-10 worst-10% ~76.3 vs FedAvg ~71.5 (**approx**) | TMLR 2023 | client-fair only |
| **GIFAIR-FL** | group+indiv | loss-spread `\|L_i−L_j\|` reg | group/indiv | λ | CIFAR-10, FEMNIST, Shakespeare | improved spread & worst-group at ~equal acc (**qualitative**) | INFORMS JDS 2023 | one λ, one notion |
| **AdaFed** | client-fair | min-norm common descent `\|f\|^γ` | worst-client | γ | CIFAR-10, FEMNIST | CIFAR-10 worst-10% 58.24 / std 4.50 beats q-FFL 47.29/5.60 (**reported**) | TMLR 2024 | one model, no group notion |
| **FedMinMax** | group-fair | minimax over groups | minimax group | ε floor | Adult, FMNIST, CIFAR-10, ACS | Adult worst-group risk ~0.25–0.27 vs FedAvg ~0.35–0.38 (**from figs**) | FAccT 2022 | needs group labels + per-group risk sharing |
| **FairFed** | group-fair agg | deviation nudge on `\|F_g−F_k\|` | EOD/SPD | β | Adult, COMPAS | Adult EOD −0.174→−0.017 (~93%↓) at 0.830 vs 0.835 acc, α=0.1 (**verified via ar5iv**) | AAAI 2023 | needs A at every client; noisy/undefined under SSG; gameable (PFAttack) |
| **FedFB** | group-fair | bilevel group reweight | DP/EO | — | Adult, COMPAS, synthetic | near-centralized FairBatch gaps (**exact table not extractable**) | arXiv 2021 | leaks per-group counts |
| **FCFL** | client+group MOO | constrained min-max, per-round LP | worst-client + group | — | Adult, eICU | lower disparity + worst-client loss vs AFL/q-FFL (**qualitative**) | NeurIPS 2021 | expensive per-round LPs; needs A at clients |
| **FAIR-FATE** | group-fair agg | fairness-gated momentum | SP/EO/EQO | β sched | COMPAS, Adult, Law, Dutch | COMPAS SP 1.00 vs 0.59, EO 0.95 vs 0.46 (ratios, σ=0.5) (**reported**) | ICCS 2023 | needs clean server val set |
| **GLocalFair** | group-fair | Gini-clustered aggregation | local+global group | — | 2 image + 1 tabular | lower EOD/SPD vs FairFed (**qualitative**) | NDSS 2024 | clustering heuristic |
| **LoGoFair** | post-hoc | Bayes-opt thresholds, local+global | DP/EO | — | Adult, ENEM, CelebA | Adult α=0.5: **0.0489 local / 0.0204 global** viol vs FCFL 0.0832/0.1479 (**verified via HTML**) | arXiv 2025 | still needs A at clients |
| **FedFACT** | post-hoc | controllable calibration | DP/EO | budget ε | tabular | controllable frontier dominating FairFed/FedFB (**no single headline**) | arXiv 2025 | post-hoc only |
| **Zhou & Goel post-proc** | post-hoc | FedAvg + local relabel | EO | — | Adult, COMPAS, PTB-XL, CXR | COMPAS ~79% EOD↓ at α=0.5 (**author-reported**) | arXiv 2025 | local fairness only |
| **Adversarial debiasing** | in-proc (local) | gradient projection vs adversary | DP/EO/EOpp | — | Adult, embeddings | gaps→~0 small acc cost | AIES 2018 | unstable under non-IID/short epochs |
| **Reductions / EG (fairlearn)** | in-proc | Lagrangian → cost-sensitive ERM | DP/EO | — | Adult, COMPAS, LSAC | provable constraint satisfaction | ICML 2018 | needs shared per-group stats |
| **Hardt threshold opt** | post-hoc | per-group thresholds LP | EO/EOpp | — | — | closed-form; AIF360/fairlearn ref | NeurIPS 2016 | local-only in FL |
| **DRO (Hashimoto)** | demographics-free | χ²-ball worst-case reweight | Rawlsian | radius | text sims | representation-disparity ↓ vs ERM | ICML 2018 | conflates hard vs minority (noise-sensitive) |
| **ARL** | demographics-free | adversary reweights on (X,Y) | Rawlsian | — | Adult, LSAC, COMPAS | worst-group AUC ↑ vs ERM/DRO (**directional only; source numbers fabricated per survey**) | NeurIPS 2020 | computational-identifiability may fail |
| **Group-DRO** | group robust | worst-group loss min | worst-group | reg | Waterbirds, CelebA, CivilComments | large worst-group acc gains | ICLR 2020 | needs group labels |
| **FedDF** | ensemble fusion | logit-avg teacher → distill student | accuracy | — | CIFAR, ImageNet, AGNews | fewer rounds, beats FedAvg/FedProx/FedMA (**qualitative**) | NeurIPS 2020 | accuracy-only; needs unlabeled server data |
| **FedBE** | ensemble agg | Bayesian model ensemble + SWA distill | accuracy | — | CIFAR | +2–9% over FedAvg; ResNet20 73.4 vs 70.2 (**reported**) | ICLR 2021 | accuracy-only |
| **AdaFair** | boosting (centralized) | cumulative-fairness AdaBoost | EO | ensemble size θ | Adult, Bank, COMPAS, KDD | Eq.Odds ~2% on KDD, ~25% better balanced err (**reported**) | CIKM 2019 | centralized; runaway weights |
| **FEAMOE** | MoE (centralized) | fairness-constrained experts + gate | DP/EO/+1 | — | HMDA + tabular | fairer at comparable acc, drift-adaptive | arXiv 2022 | centralized; feature-space gate |
| **pFedMoE** | FL MoE | generalist+specialist gate | accuracy/personalization | — | vision | SOTA personalized acc | arXiv 2024 | not fairness; personalization gate |
| **FeDABoost** | FL boosting | perf-based agg weights | client-perf | — | MNIST, FEMNIST, CIFAR | reduced cross-client variance (**no table**) | arXiv 2025 | client-perf only, no group notion |
| **FedMGDA+ / MOO surgery (PCGrad, CAGrad, FedLF, FedGF)** | MOO | min-norm / gradient-surgery Pareto | Pareto-stationary | — | vision/tabular | local Pareto-stationarity of gradients | 2020–2024 | stationarity of gradients ≠ group-metric optimality |

**Notes on provenance.** Cells marked **verified** were confirmed by the survey against primary sources (arXiv Table 1 / ar5iv / HTML). Cells marked **reported / approx / qualitative** are author-reported or summarizer-extracted and should be treated as directional. The ARL AUC figures were flagged by the survey as **fabricated by a PDF summarizer**; only the directional claim is sourced.

### 2.4 Multi-objective optimization (MOO) angle

MOO methods (FedMGDA+, PCGrad, CAGrad, FedLF, FedGF, FCFL) find a common descent direction that is Pareto-stationary in *gradient* space. They do **not** guarantee optimality of the *group-metric* Pareto front, are expensive (per-round QP/LP), and still commit to one operating point per run. FedGMA's convex-hull-over-rules view is a cheaper, closed-form alternative that traces the frontier post-hoc (§4.4) without online QP.

### 2.5 The open gaps FedGMA targets

- **G1 — single op-point / per-dataset re-tuning** (survey key finding #4). Every method fixes one frontier point.
- **G2 — client-fair vs group-fair non-coexistence** (FedMinMax lemma).
- **G3 — group aggregators break under SSG / noisy / adversarial fairness stats** (FairFed local metric undefined under single-group-per-client; PFAttack 2024 spoofs fairness stats).
- **G4 — no method offers a no-regret / no-worse-than-best-baseline guarantee at the aggregation step.**

### 2.6 Citations (primary)

- q-FFL/q-FedAvg — Li, Sanjabi, Beirami, Smith — ICLR 2020 — https://arxiv.org/abs/1905.10497
- AFL — Mohri, Sivek, Suresh — ICML 2019 — https://arxiv.org/pdf/1902.00146
- Ditto — Li, Hu, Beirami, Smith — ICML 2021 — https://arxiv.org/pdf/2012.04221
- FedMinMax — Papadaki, Martinez, Bertran, Sapiro, Rodrigues — FAccT 2022 — https://arxiv.org/abs/2201.08304
- FairFed — Ezzeldin, Yan, He, Ferrara, Avestimehr — AAAI 2023 — https://arxiv.org/abs/2110.00857
- FedFB — Zeng, Chen, Lee — arXiv 2021 — https://arxiv.org/abs/2110.15545
- FCFL — Cui, Pan, Liang, Zhang, Wang — NeurIPS 2021 — https://arxiv.org/abs/2108.08435
- PropFair — Zhang, Malekmohammadi, Chen, Yu — TMLR 2023 — https://arxiv.org/abs/2202.01666
- GIFAIR-FL — Yue, Nouiehed, Al Kontar — INFORMS JDS 2023 — https://arxiv.org/abs/2108.02741
- FAIR-FATE — Salazar, Fernandes, Araujo, Abreu — ICCS 2023 — https://arxiv.org/abs/2209.13678
- AdaFed — Hamidi, Yang — TMLR 2024 — https://arxiv.org/abs/2401.04993
- GLocalFair — Meerza et al. — NDSS 2024 — https://arxiv.org/abs/2401.03562
- LoGoFair — arXiv 2025 — https://arxiv.org/pdf/2503.17231
- FedFACT — arXiv 2025 — https://arxiv.org/html/2506.03777v1
- Post-processing Fair FL — Zhou, Goel — arXiv 2025 — https://arxiv.org/html/2501.15318v1
- Adversarial debiasing — Zhang, Lemoine, Mitchell — AIES 2018 — https://dl.acm.org/doi/10.1145/3278721.3278779
- Reductions — Agarwal et al. — ICML 2018 — https://arxiv.org/abs/1803.02453
- Hardt threshold opt — Hardt, Price, Srebro — NeurIPS 2016 — https://arxiv.org/abs/1610.02413
- DRO — Hashimoto, Srivastava, Namkoong, Liang — ICML 2018 — https://arxiv.org/abs/1806.08010
- ARL — Lahoti et al. — NeurIPS 2020 — https://alexbeutel.com/papers/NeurIPS-2020-fairness-without-demographics.pdf
- Group-DRO — Sagawa, Koh, Hashimoto, Liang — ICLR 2020 — https://arxiv.org/abs/1911.08731
- FedDF — Lin et al. — NeurIPS 2020 — https://arxiv.org/abs/2006.07242
- FedBE — Chen, Chao — ICLR 2021 — https://openreview.net/pdf?id=dgtpE6gKjHn
- AdaFair — Iosifidis, Ntoutsi — CIKM 2019 — https://arxiv.org/abs/1909.08982
- Hedge / multiplicative weights — Freund, Schapire 1997; Arora, Hazan, Kale 2012 (survey of MW); Cesa-Bianchi & Lugosi, *Prediction, Learning, and Games*, 2006
- Survey of group fairness in FL — Rafi et al. — arXiv 2410.03855 — https://arxiv.org/pdf/2410.03855

---

## 3. SOTA target numbers to beat (per dataset)

**Read this first.** These are the *literature* targets. The repo currently evaluates on `adult, compas, mimic, eicu` (not Diabetes-130) and its committed results are degenerate for every method (AUROC ≈ 0.50). So the operational target for week 1 is not "beat SOTA" — it is **reproduce a non-degenerate baseline that lands in the operating ranges below.** Only then do the "beat" columns apply. Confidence tags: **[V]** verified against primary source, **[R]** author-reported, **[A]** approximate/derived, **[O]** open (thin FL precedent).

### 3.1 Adult (UCI / ACS-Income), sensitive attribute = sex (secondary: race)

| Metric | Operating range | SOTA reference point to beat | Provenance |
|---|---|---|---|
| Accuracy | 0.83–0.85 | FairFed 0.830 (vs FedAvg 0.835) at α=0.1 | **[V]** Ezzeldin 2023 |
| EO gap (EOD) | ≤ 0.02 | FairFed −0.174 → **−0.017** (~93%↓) | **[V]** Ezzeldin 2023 |
| Local / global fairness violation | ≤ 0.05 / ≤ 0.02 | LoGoFair **0.0489 / 0.0204** at α=0.5 | **[V]** LoGoFair 2025 |
| Worst-group risk | ≤ 0.27 | FedMinMax ~0.25–0.27 (vs FedAvg ~0.35–0.38) | **[A]** Papadaki 2022 |
| **FedGMA success bar** | — | **EO gap ≤ 0.02 AND worst-group-F1 ≥ best single baseline, at accuracy ≥ 0.83, in ONE run with no per-dataset knob tuning** | design target |

### 3.2 COMPAS, sensitive attribute = race

| Metric | Operating range | SOTA reference point to beat | Provenance |
|---|---|---|---|
| Accuracy | 0.65–0.68 | FairFed 0.672 | **[R]** Ezzeldin 2023 |
| EO gap (EOD) | ≤ 0.03 | FairFed −0.065 → **−0.057** (~50%↓); Zhou&Goel ~79%↓ at α=0.5 | **[R]** Ezzeldin 2023; Zhou&Goel 2025 |
| SP ratio / EO ratio (→1.0 fair) | ≥ 0.95 | FAIR-FATE **SP 1.00, EO 0.95** (vs FedAvg 0.59 / 0.46), σ=0.5 | **[R]** Salazar 2023 |
| **FedGMA success bar** | — | **EO gap ≤ 0.03 at accuracy ≥ 0.66, worst-group-F1 ≥ best single baseline, one run** | design target |

### 3.3 Diabetes-130 (UCI readmission), sensitive attribute = race / gender / age

| Metric | Operating range | Reference point | Provenance |
|---|---|---|---|
| Accuracy (binary <30d readmit) | 0.60–0.64 | centralized tabular baselines | **[A]** general lit |
| AUROC | 0.64–0.68 | centralized gradient-boosted / MLP baselines | **[A]** general lit |
| EO / DP gap | **no canonical FL-fairness number** | — | **[O]** |
| **FedGMA opportunity** | — | **Establish the FL-fairness benchmark: report FedAvg baseline + FedGMA at EO/DP gap ≤ 0.05 while holding AUROC ≥ 0.65 and worst-group-F1 ≥ best single baseline** | contribution |

**Diabetes-130 is a deliberate contribution surface, not a place to chase an existing headline.** It has thin FL-fairness precedent; the field mostly uses Adult/COMPAS/ACS. Establishing a clean FedAvg→FedGMA benchmark here (heavy class imbalance, three candidate sensitive attributes, natural non-IID by hospital/payer) is itself publishable and is where FedGMA's demographics-free channel and SSG robustness (§4.5) are most differentiating. **Provenance caveat:** do not cite a specific Diabetes-130 fairness SOTA number — none is reliable; report your own measured baseline.

**Reporting discipline (non-negotiable).** Because repo numbers are known-unreliable and prior results were fabricated/degenerate (memory + §1), **every** number FedGMA reports must be (a) measured on real Adult/COMPAS/Diabetes-130, (b) accompanied by its own faithful re-run of each baseline (not literature values transplanted as "our baselines"), and (c) reported with seed variance and the surrogate↔test correlation (§4.6, §5).

---

## 4. THE CHOSEN METHOD — FedGMA (full specification)

### 4.1 Name and thesis

**FedGMA — Federated Gated Mixture-of-Aggregators with No-Regret Fair Blending.**

> **Thesis (one sentence).** Fair-FL aggregation-rule selection can be cast as *full-information online convex optimization* through a server-side **counterfactual surrogate** `L_t(w)` that is **convex in the aggregation weight vector `w`** and **evaluable for every candidate rule without deploying it** — yielding the first federated fair-aggregator that is *provably no worse than the best single baseline chosen in hindsight* (regret `O(√(T ln M))`), coupled to slow-timescale dual ascent for `O(1/√T)` constrained group-fairness satisfaction.

### 4.2 Novelty — what is new vs. prior art

| Claim | Prior closest | Distinction |
|---|---|---|
| **Full-information OCO over aggregation RULES via a counterfactual convex surrogate** | q-FFL (one knob); FedMGDA+ (min-norm over *gradients*); Auto-FedAvg/FedExP (adaptive server LR, not rule selection); "learning to aggregate" | The surrogate `L_t(w) = Σ_k w_k ℓ^util_k + Σ_c λ_c |Σ_k w_k Δr^{(c)}_k|` is **linear + abs-of-linear ⇒ convex in `w`** and can be scored at *every* expert's `w^(m)` from client reports alone. This converts gating into full-information OCO — the specific object I could not find published. **This is the paper's contribution.** |
| **No-worse-than-best-component guarantee at the aggregation step** | single-model fair-FL (no such guarantee); FedBE/FedDF (ensemble, no regret bound) | Hedge over the M rules + Jensen (convexity of `L_t`) gives `Σ_t L_t(w_t) − min_m Σ_t L_t(w^{(m)}) ≤ √(T ln M / 2)`. |
| **Two-timescale primal–dual coupling (fast gate, slow duals)** | FedMinMax (ascent on group weights); Lagrangian fairness | Fast Hedge over rules + slow dual ascent on `(gap_c − ε_c)` = saddle-point for `min_{conv(experts)} max_λ` Lagrangian ⇒ `gap_c ≤ ε_c + O(1/√T)`. |
| **Coverage-weighted demographics-free conditioning** | FairFed (undefined under SSG); GLocalFair (Gini clustering) | Gate fuses measured group gaps (weighted by group coverage) with an always-available demographics-free proxy, shifting α-mass to the DRO/AFL experts when coverage → 0. |
| **Cross-expert disagreement as poisoning defense** | FairFed/FedFB (trust client stats; PFAttack breaks them) | `Var_m(w^{(m)}_k)` bounds how far one falsified report can move the blend ⇒ graceful-degradation cap. |

**Honest scope of the claim.** The reachable set of the blend is exactly `conv(experts)`. FedGMA therefore **does not** claim to strictly Pareto-dominate every baseline on every metric; it claims to **self-tune to the best convex mixture of the members with a regret certificate**, i.e., robustness to knob-misspecification + no-regret. Framing it as "dominates all baselines" would be false and reviewers would (correctly) reject it.

### 4.3 Borrowed components (which algorithm each comes from)

| Component | Borrowed from | Role in FedGMA |
|---|---|---|
| Expert bank (M weight-vector rules) | FedAvg; q-FFL (loss^q); AFL (minimax); FairFed (inverse-EO nudge); Group-DRO/FedMinMax (softmax over per-group risk) | The M vertices whose convex hull the gate searches |
| Hedge / exponentiated-weights | Freund–Schapire 1997; Cesa-Bianchi–Lugosi 2006; Arora–Hazan–Kale 2012 | No-regret update → the guarantee |
| Slow dual ascent | FedMinMax / Lagrangian constrained fairness | Turns fixed penalty into constraint chasing; feeds the gate state |
| Demographics-free bias signal | ARL (Lahoti 2020) + DRO (Hashimoto 2018) + repo DFBD net | Channel-B detection under SSG/no-attribute |
| Coverage-weighted fusion + disagreement cap | FairFed weakness analysis + PFAttack threat model | Robustness / SSG survival |
| MoE gate primitive (optional learned variant) | pFedMoE / FEAMOE | `g_φ(s_t)` routes over *rules* by bias state (re-purposed) |
| Output-space distillation (**optional ablation, NOT spine**) | FedDF / FedBE | Hull-escape correction — kept out of core for cost/correlation reasons |

### 4.4 Aggregation math (the core rule)

**Setup.** Round `t`, `K` participating clients, `M` experts. Client `k` reports: update `δ_k`, sample count `n_k`, validation loss `ℓ^util_{t,k}`, per-group confusion counts / per-group risks, and demographics-free proxies `{loss_drift, delta_norm, ece_proxy}`. `Δ_{K−1}` is the client simplex.

**1. Experts.** Each expert `m` builds `w^{(m)}_t ∈ Δ_{K−1}` (all normalized to sum 1):

- **FedAvg:** `w_k ∝ n_k`
- **q-FFL (real):** `w_k ∝ p_k · (ℓ^util_{t,k})^q`, e.g. `q ∈ {0.5, 2, 5}` as separate experts *(fixes the repo's inverted `1/(loss²+ε)`)*
- **AFL / minimax (real):** `w_k ∝ exp(ℓ^util_{t,k} / τ)` — **up-weights** the worst client *(fixes the repo's inverted `1/(loss+ε)`)*
- **FairFed:** `w_k ∝ nudge on |F_global − F_k|` (inverse-EO-gap)
- **Group-DRO / FedMinMax:** `w_k ∝ softmax_k(worst-group-risk_k / τ)`

**2. Bias state** `s_t ∈ ℝ^d`:
`s_t = [ EO_gap, FPR_gap, SP_gap, worst_group_F1, mean(ℓ), var(ℓ), DFBD_round_score, λ_eo, λ_fpr, λ_sp, group_coverage_frac, t/T ]`.

**3. Gate** `α_t ∈ Δ_{M−1}`:
- **Theory variant (GMA-Hedge):** `α_t = p_t`, the exponential-weights distribution.
- **Learned variant (GMA-Gate, ablation):** `α_t = (1−ρ) g_φ(s_t) + ρ p_t`, `ρ` anneals 1→0.

**4. Blend (core rule):**
```
w_t = Σ_m α_{t,m} · w^{(m)}_t
```
Since each `w^{(m)}_t ∈ Δ_{K−1}` and `α_t ∈ Δ_{M−1}`, **`w_t ∈ Δ_{K−1}` automatically — a valid aggregator with no renormalization.**

**5. Robust cap:** `d_k = Var_m(w^{(m)}_{t,k})`; `cap_k = c0 / (1 + κ·d_k)`; `w_t ← Clip(w_t, cap_k)` then renormalize (iterative capper).

**6. Server update:** `θ_{t+1} = θ_t + η_s · Σ_k w_{t,k} · δ_k` (optional server momentum on the blended direction).

**7. Slow duals:** `λ_{c,t+1} = clip(λ_{c,t} + η_λ·(gap_c(w_t) − ε_c), 0, λ_max)` for `c ∈ {eo, fpr, sp}`.

**8. Fast gate (full-information OCO). The load-bearing object:**
```
L_t(w) = Σ_k w_k · ℓ^util_{t,k}   +   Σ_c λ_{c,t} · | Σ_k w_k · Δr^{(c)}_{t,k} |
```
where `Δr^{(c)}_{t,k}` is client `k`'s **signed** contribution to group-`c` risk gap (FedMinMax-style per-group empirical risk difference). The first term is **linear** in `w`; the second is **|linear| ⇒ convex**. So `L_t` is convex in `w` and **evaluable at every `w^{(m)}_t` from reports without deploying any candidate.**

Per-expert loss `ℓ_{t,m} = L_t(w^{(m)}_t)`, scaled to `[0,1]`. **Hedge update:**
```
p_{t+1,m} ∝ p_{t,m} · exp(−η_g · ℓ_{t,m}),   η_g = √(8 ln M / T)
```
**Regret.** By convexity + Jensen, `L_t(w_t) = L_t(Σ_m p_{t,m} w^{(m)}_t) ≤ Σ_m p_{t,m} L_t(w^{(m)}_t) = ⟨p_t, ℓ_t⟩`. Hedge on `ℓ_t` gives `Σ_t ⟨p_t, ℓ_t⟩ − min_m Σ_t ℓ_{t,m} ≤ √(T ln M / 2)`. Therefore
```
Σ_t L_t(w_t) − min_m Σ_t L_t(w^{(m)}_t) ≤ √(T ln M / 2)
```
— **the blend is asymptotically no worse than the best single aggregation rule in hindsight (on the surrogate).**

**Learned variant** additionally steps `φ` to minimize `L_t(Σ_m g_φ(s_t)_m w^{(m)}_t)` (differentiable in `α`) with cross-entropy anchoring to `p_{t+1}`; it inherits the regret bound up to an imitation gap.

### 4.5 Bias-detection mechanism (two-channel, coverage-fused)

- **Channel A (measured, when A present at a client):** clients report per-group confusion counts; server computes global EO/FPR/SP gaps and worst-group F1, **trusted in proportion to each round's group-coverage fraction**.
- **Channel B (demographics-free, always available):** the DFBD network consumes `{loss_drift, delta_norm, ece_proxy}` (no sensitive attribute) → per-client bias tilt → round-level bias score.
- **Fusion:** the two enter `s_t` with a coverage weight. When coverage → 0 (SSG), the gate relies on Channel B and shifts α-mass to the demographics-free experts (AFL/DRO) — closing **G3**.
- **Detection ⇒ mitigation trigger:** raises `λ_fair`, sets `bias_mitigation_mode`, requests an extra client fairness epoch (`get_fairness_config`).
- **Poisoning guard:** a client whose self-reported fairness stat is inconsistent with its DFBD proxy / reported loss is down-weighted in the FairFed/DRO experts and capped via cross-expert disagreement (`Var_m w^{(m)}_k`) — so PFAttack-style falsified reports cannot dominate the blend.

> **Critical correctness note.** The repo's DFBD net (`faircare_fl.py:186`) **is never trained** — `dfbd_optimizer.step()` is never called (grep-confirmed), so Channel B currently emits **random tilts**. **Do not feed a random signal into `s_t` and call it demographics-free detection.** Either (a) train the DFBD net online against a held-out worst-group signal, or (b) demote Channel B to a **fixed, deterministic prior** (e.g., normalized `loss_drift`) until it is trained. This is a hard precondition, not a nicety.

### 4.6 Training loop

```
SERVER init: experts (weight builders), Hedge p_0 = uniform over M, optional g_φ (Adam),
             duals λ = 0, DFBD net (trained OR demoted to fixed prior), server momentum buffer.
             Broadcast θ_0 and fairness_config.

For round t = 1..T:
  CLIENTS (parallel): receive θ_t, fairness_config. Local train (+ optional local debiaser
      / extra fairness epoch if bias_mitigation_mode). Report:
      δ_k, n_k, ℓ^util_k, per-group confusion counts/risks (secure-agg / DP where enabled),
      proxies {loss_drift, delta_norm, ece_proxy}, worst_group_f1.
      Sensitive attributes NEVER leave the client; attribute-free clients report only proxies.
  SERVER:
   (a) Build M expert vectors w^{(m)}_t          [_compute_component_weights — FIXED members]
   (b) Measured gaps (_compute_fairness_metrics) + DFBD round score (_compute_advanced_tilts)
       → assemble s_t (coverage-weighted fusion)
   (c) α_t = gate(s_t)  (Hedge p_t and/or g_φ);   BLEND w_t = Σ_m α_{t,m} w^{(m)}_t
   (d) Robust disagreement cap on w_t              [_postprocess iterative capper]
   (e) θ_{t+1} = θ_t + η_s Σ_k w_{t,k} δ_k         [weighted_average_weights + apply_model_delta]
   (f) Slow dual ascent on (gap_c − ε_c)           [_update_dual_variables]
   (g) Fast gate: ℓ_{t,m} = L_t(w^{(m)}_t) ∀m; Hedge step on p; (learned: one Adam step on φ)
   (h) Set bias_mitigation_mode / λ_fair; update fairness_config for next broadcast
   (i) Poisoning check: down-weight FairFed/DRO experts for inconsistent clients

Deployment: freeze α and post-hoc sweep it to read off any point on conv(experts) — the traced frontier.
Two-timescale: η_g ≫ η_λ; η_g from the regret schedule; ρ anneals 1→0.
```

### 4.7 Exactly how to implement it in `faircare/algos/faircare_fl.py`

**Live seam (where FedGMA plugs in).** `faircare/core/server.py:181` calls `self.aggregator.compute_weights(client_stats)` → `weighted_average_weights` → `apply_model_delta(..., lr=server_lr)`. This is **weight-space** and is the *only* live path. FedGMA replaces the **body of `FairCareFLWrapper.compute_weights`** (`faircare_fl.py:1282`). *(The `FairCareFLAggregator.aggregate()` path at `:236`, used by `run_experiments.py:394`, is the orphaned distillation path that produced the committed degenerate results — do not build on it for the spine; migrate `run_experiments.py` to the same wrapper seam so both pipelines are identical.)*

| Step | Repo hook | Current state | Change |
|---|---|---|---|
| Expert bank | `_compute_component_weights` `:1372` | **DEAD (no call site)** | **Wire it into `compute_weights`** (`:1350` currently calls `_compute_optimal_weights` instead). **Fix members:** `qffl` `:1402` `1/(loss²+ε)` → `p_k·loss^q` (add q∈{0.5,2,5}); `afl` `:1409` `1/(loss+ε)` → `exp(loss/τ)` (real minimax up-weight); `fedprox` `:1406` (clone of fedavg) → **replace with group-DRO** `softmax(worst_group_risk/τ)`. |
| Gate | `gate_network` `:1257` `Linear(5,16)→ReLU→Linear(16,5)→Softmax` | **DEAD (never invoked)** | Becomes `g_φ`: widen input `5→d=len(s_t)`, output `5→M`. Add **Hedge buffer `p_t`** as a server tensor + the α-blend. Start with **GMA-Hedge only** (no learned gate) for the spine. |
| Bias state | `_compute_fairness_metrics` `:786`, `_compute_advanced_tilts` `:865`, duals `:150` | metrics OK; DFBD **untrained** | Assemble `s_t`; coverage-weight measured gaps; **train DFBD (`dfbd_optimizer.step()`, `:186`) or demote to fixed prior**. |
| Blend + apply | `_compute_optimal_weights`/`_compute_enhanced_weights` `:902/:921`; `weighted_average_weights`; `apply_model_delta` (server) | live | Replace the combine step with `w_t = Σ_m α_m w^{(m)}`; keep server step. |
| Duals | `_update_dual_variables` `:753`; `λ_eo/fpr/sp` `:150` (`ε=0.015/0.015/0.02`, `dual_lr=0.005`) | live | Reuse verbatim as the **slow timescale** (set `η_λ ≪ η_g`). |
| Robust cap | `BaseAggregator._postprocess` (`aggregator.py` iterative capper) | live | Per-client `cap_k` from `Var_m(w^{(m)}_k)`. |
| Surrogate + Hedge | — | new | Add small server method computing `L_t(w^{(m)})` ∀m and the multiplicative-weights step. ~150–250 LOC total. |
| Distillation | `_perform_distillation` `:1106` | **stub returns `{'distill_loss': 0.0}`** | **Leave OFF for the spine.** Only touch as the optional hull-escape ablation (§5), and only after a probe-set exists. |

**Config bug.** Memory flags a config import bug and enhanced-config feature flags. Fix the config path and gate the enhanced-config flags **before** trusting any run; otherwise `_compute_enhanced_weights` (`:921`) silently changes behavior.

---

## 5. Ablation + experiment plan

**Datasets.** Adult (sex; race secondary), COMPAS (race), Diabetes-130 (race/gender/age). Non-IID via Dirichlet `α ∈ {0.1, 0.5, 5}`. Group-access settings **ESG / PSG / SSG** (equal / partial / single-group-per-client). K=10 clients (repo default), scale to 50/100 for the AFL-degradation stress test.

**Seeds & stats.** ≥ 5 seeds per cell (10 for headline). Report mean ± std. Significance: **paired Wilcoxon signed-rank** across seeds for FedGMA vs each baseline; **Holm–Bonferroni** correction across the metric family; report effect sizes, not just p-values.

**Metrics.** Accuracy, macro-F1, AUROC, **worst-group-F1**, EO gap (EOD), FPR gap, SP gap, worst-group risk; plus **cumulative surrogate regret** and **empirical `√(ln M / T)` regret rate**.

**Ablations (each isolates one claim):**

1. **Expert-bank leave-one-out** (−FairFed, −AFL, −DRO, −q-FFL, −DFBD-expert). Collapsing to a single expert **must** recover that exact baseline (faithfulness sanity — this also catches the mislabel regression).
2. **Gate variant:** static-uniform α vs **GMA-Hedge** vs GMA-Gate (learned) vs **oracle best-fixed-expert-in-hindsight** (upper bound). Verify empirical regret matches `√(ln M / T)` and learned ≥ Hedge.
3. **Bias-state features:** ± DFBD demographics-free score; ± duals in `s_t`.
4. **Robustness sweep:** PFAttack-style falsified fairness reports at `ε ∈ {0, 0.1, 0.2, 0.4}`, with/without disagreement cap — expect graceful degradation vs FairFed/FedFB collapse.
5. **Heterogeneity × group access:** Dirichlet `α` × {ESG, PSG, SSG} — expect FedGMA to **hold in SSG** where FairFed's local metric is undefined.
6. **Two-timescale stability:** sweep `η_g/η_λ` and `ρ`-annealing → map the stable region.
7. **Frontier-in-one-run:** post-hoc sweep frozen α; plot accuracy vs EO-gap / worst-group, overlaid on a grid of single-baseline runs — expect **one FedGMA run to trace what took N baseline runs.**
8. **(Load-bearing) Surrogate validity:** Spearman correlation between `L_t` and held-out {EO gap, worst-group-F1, accuracy} across rounds. **This must be in the main body** — if the surrogate does not correlate with test metrics, the regret theorem is decorative. Better to learn this in week 2 than in rebuttal.
9. **(Optional) GMA vs GMA-Distill:** distillation on/off to quantify hull-escape gain vs compute cost.

**Baseline re-implementation requirement.** Run FedAvg, real q-FFL, real AFL, FairFed, FedMinMax/group-DRO **in-repo** on the same splits/seeds. Do not transplant literature numbers as "our baselines."

---

## 6. Risks & fallbacks

| Risk | Severity | Mitigation / fallback |
|---|---|---|
| **Pipeline degeneracy persists** (constant-class predictor, AUROC ≈ 0.5) — *the dominant risk; nothing is measurable until fixed* | Blocking | De-degenerate on real Adult/COMPAS/Diabetes-130 first (§7.1). Fallback: if group-fairness signal is too weak on tabular, add ACS-Income (larger, cleaner group structure) as a fourth benchmark. |
| **Surrogate ↔ test decorrelation** (regret bound is w.r.t. `L_t`, not held-out metrics) | High | Ablation #8 in main body; use a server probe set when available; report both GMA-Hedge and GMA-Gate (bracket the risk). Fallback: if correlation is weak, reframe the paper around the *self-tuning* empirical result and demote the theorem to a motivating bound. |
| **Reachability limit** — blend only reaches `conv(experts)`; if all experts share a bias no α fixes it | Medium | Include the demographics-free DRO member; keep GMA-Distill as the (optional) hull-escape. Frame claim honestly (§4.2). |
| **DFBD net untrained → random Channel B** | High (correctness) | Train `dfbd_optimizer` or demote to a fixed deterministic prior; never feed random tilts into `s_t`. |
| **Members are correlated reweightings of one trajectory** | Medium | Widen the hull with genuinely diverse *correctly-implemented* members (real q∈{0.5,2,5}, real AFL, group-DRO, FairFed) — the hybrid borrow from BCPE-FL's "spread the knob" idea. Do **not** rely on ensemble-cancellation claims. |
| **Two-timescale oscillation** (FAIR-FATE failure mode) | Medium | `η_g ≫ η_λ` from the regret schedule; server momentum on the blended direction; `ρ` annealing. |
| **Privacy surface** — FairFed/DRO experts need per-group counts | Medium | Secure-aggregate / DP-noise the group stats; provide a privacy-only mode that drops group-reporting experts and runs on DFBD alone. |
| **Config import bug / enhanced-config flags silently change behavior** | Medium | Fix config path and gate feature flags before any run (memory-flagged). |
| **Diabetes-130 has no FL-fairness SOTA to "beat"** | Low | Reframe as establishing the benchmark (§3.3); this is a contribution, not a gap. |
| **Compute** | Negligible | Tiny MLPs, 10 clients, tabular — a T4 is over-provisioned. Compute is *not* the constraint; correctness is. |

---

## 7. Prioritized implementation checklist

**Ordering is strict: no number is reportable until 7.1–7.3 pass.**

### 7.1 De-degenerate + real data (BLOCKING, method-independent)
- [ ] Confirm a plain **FedAvg** baseline yields **non-degenerate worst-group-F1** (not gaps==0 via constant predictor) on **Adult / COMPAS / Diabetes-130**; AUROC must exceed ~0.6, not ~0.5.
- [ ] Fix the config import bug; gate enhanced-config feature flags (`_compute_enhanced_weights` `:921`).
- [ ] Unify the two divergent seams: migrate `run_experiments.py:394` off `aggregate()` onto the `compute_weights` wrapper path so both pipelines are identical.

### 7.2 Implement faithful baselines (guarantee is meaningless against strawmen)
- [ ] Fix `qffl` `faircare_fl.py:1402` `1/(loss²+ε)` → `p_k·loss^q` (add q∈{0.5,2,5}).
- [ ] Fix `afl` `:1409` `1/(loss+ε)` → `exp(loss/τ)` (real minimax up-weight).
- [ ] Replace `fedprox` clone `:1406` with a **group-DRO** member `softmax(worst_group_risk/τ)`.
- [ ] Faithfulness test: single-expert collapse recovers each baseline exactly (ablation #1).

### 7.3 Validate the surrogate (load-bearing theory hook)
- [ ] Implement `L_t(w)` and compute Spearman corr with held-out {EO gap, worst-group-F1, accuracy} across rounds. If weak, invoke §6 fallback **before** building the full gate.

### 7.4 Build the FedGMA spine (weight-space only)
- [ ] Wire `_compute_component_weights` `:1372` into `compute_weights` `:1282` (it is currently dead).
- [ ] Add Hedge buffer `p_t` + multiplicative-weights step (`η_g = √(8 ln M/T)`).
- [ ] Blend `w_t = Σ_m p_{t,m} w^{(m)}_t`; reuse `_postprocess` capper with `cap_k` from `Var_m(w^{(m)}_k)`.
- [ ] Couple slow duals `_update_dual_variables` `:753` (`η_λ ≪ η_g`) into `L_t` and `s_t`.
- [ ] Assemble `s_t`; **train DFBD (`:186`) or demote Channel B to a fixed prior** (never random).

### 7.5 Ablations + stats (§5)
- [ ] Gate variants incl. oracle upper bound; verify empirical regret ≈ `√(ln M/T)`.
- [ ] PFAttack robustness sweep with/without disagreement cap.
- [ ] Heterogeneity × {ESG,PSG,SSG}; confirm SSG survival.
- [ ] Frontier-in-one-run plot vs N baseline runs.
- [ ] ≥5 seeds; paired Wilcoxon + Holm–Bonferroni; report effect sizes.

### 7.6 Optional (only if 7.1–7.5 land and time allows)
- [ ] Learned gate `g_φ` (widen `gate_network` `:1257` input→d, output→M; anchor to `p_{t+1}`).
- [ ] GMA-Distill hull-escape: implement `_perform_distillation` `:1106` (currently a zero-returning stub) on a group-balanced probe — **only after a probe-set exists**; keep it an ablation, not the spine.
- [ ] Post-hoc LoGoFair/FedFACT-style calibration layer for exact-budget EO/DP.

### 7.7 Reporting discipline
- [ ] Every number measured on real data with own baseline re-runs (no transplanted literature values).
- [ ] Frame the claim as **self-tuning to best-baseline-in-hindsight + regret**, never as Pareto domination.
- [ ] Report the surrogate↔test correlation and the regret curve in the main body.

**Load-bearing files:** `D:\mango_git\mango\faircare\algos\faircare_fl.py` (experts `:1372`; mislabeled `qffl` `:1402` / `afl` `:1409` / `fedprox` `:1406`; gate stub `:1257`; distill stub `:1106`; untrained DFBD `:186`; duals `:150`/`:753`; live `compute_weights` `:1282`), `D:\mango_git\mango\faircare\core\server.py:181` (weight-space seam), `D:\mango_git\mango\faircare\experiments\run_experiments.py:394` (orphaned `aggregate()` path that produced the degenerate committed results), `D:\mango_git\mango\results\full_evaluation\comprehensive_analysis.csv` (current degenerate baselines; note: contains adult/compas/mimic/eicu, **not** Diabetes-130 — the benchmark set must be reconciled with §3).