# paper/ethics_broader_impact.md

## Ethics & Broader Impact (Healthcare FL + Fairness)

**Human data**: If using real clinical data (e.g., MIMIC-IV/eICU), ensure CITI training/DUA, de-identification, and access controls. Document consent basis and governance. :contentReference[oaicite:11]{index=11}

**Risks**: Residual bias, distribution shift, group undercoverage, calibration drift, and privacy leakage via updates.

**Mitigations**:
- Multi-level fairness training (client/group), continuous monitoring, and alarms.
- Secure aggregation for summaries; optional differential privacy for fairness stats (configurable).
- Post-deployment audits, drift detection, and retraining schedule.
- Transparent reporting: per-group metrics, worst-case client performance, compute budget.

**Dual use**: Limit export of models if they encode site quirks; consider DP if republishing.

**Limitations**: Fairness definitions may conflict; sensitive attributes may be noisy/missing; surrogate gradients approximate true fairness.

