# paper/neurips_checklist.md

This repository includes the artifacts to satisfy the NeurIPS paper checklist (reproducibility/ethics/limitations). **IMPORTANT**: The checklist itself must be included in the paper per NeurIPS rules (desk-reject otherwise). :contentReference[oaicite:12]{index=12}

- **Data**: Sources, licenses, access requirements (MIMIC/eICU need CITI/DUA). :contentReference[oaicite:13]{index=13}
- **Code**: Training/eval scripts; seeds; pinned deps; compute; instructions; anonymized artifact (see `scripts/make_anonymous_artifact.py`). :contentReference[oaicite:14]{index=14}
- **Theoretical claims**: Assumptions & proofs in `paper/proofs_appendix.tex`.
- **Experimental results**: Multi-seed + statistical tests (`paper/tables.py`).
- **Ethics**: `paper/ethics_broader_impact.md`. Ethics guidelines inform this content. :contentReference[oaicite:15]{index=15}
- **Anonymity**: All links and code must be anonymous at submission. :contentReference[oaicite:16]{index=16}

