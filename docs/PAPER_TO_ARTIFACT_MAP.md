# IEEE manuscript claim-to-artifact crosswalk

This file maps the attached draft `IEEE_Thuy_v1.docx` (five technical pages and a separate editorial-support page) to frozen source artifacts. Numeric checks run with `python scripts/verify_paper_tables.py`.

| Manuscript item | Source within this repository | Scope / comment |
| --- | --- | --- |
| Table I: A/B/C/D configurations | `src/twitter/frozen_twitter1516_pipeline.py` and `src/external/external_validation_pipeline.py` | Features are not identical across all protocols; external study is harmonized |
| Table II: Dataset roles/sizes | `results/twitter_covid19/FROZEN_EXTERNAL_PROTOCOL.json`, `results/twitter1516_sensitivity/text_audit_summary.json` | Twitter15 1490, Twitter16 818, COVID19 400, 216 audit cases |
| Table III: Tree-policy robustness | `results/twitter1516_sensitivity/metrics_by_policy.csv`, `paired_C_D_bootstrap.csv` | Matched threshold C/D; multiple-policy exploratory analysis |
| Table IV: COVID19 external | `results/twitter_covid19/external_summary_metrics.csv`, `paired_bootstrap.csv`, `ontology_placebo_retrained.csv` | **Use D_shared, not row D**, for matched-threshold comparison; D_shared MCC −0.070195 |
| Table V: Automatic audit | `results/ontology216_automatic/AUTOMATED_AUDIT_REPORT.md`, `results/ontology216_faithfulness/reports/historical_prediction_reconstruction_audit.csv`, `EXPERIMENT_MANIFEST.json` | 216/216 source-tree checks; 16 saved CatBoost models; artifact reload |
| Table VI: Six-rule local sensitivity | `results/ontology216_faithfulness/reports/rule_faithfulness_summary.csv` | 1296 interventions, 29 decision flips; not independent causal attribution |
| Fig. 1: Pipeline | `paper/figures/pipeline.png` | Conceptual diagram |
| Fig. 2: Full tree difference/CI | `paper/figures/robustness_ci.png` and `results/twitter1516_sensitivity/paired_C_D_bootstrap.csv` | Plotted data reflects frozen policy comparison |
| PHEME related study background | `results/pheme_v16/summary_by_config.csv`, `metrics_per_event.csv` | v16 LOEO, not untouched validation |
| Audit scope: SHACL | `results/ontology216_faithfulness/provenance/ontology216_shapes.ttl`, `EXPERIMENT_MANIFEST.json` | SHACL shapes generated; **engine not executed** in recorded run |
| RDF, 60,930 triples | Full `all_216_case_provenance.ttl` available in **private research archive** | Public release intentionally omits case-level source references and raw data |

**Corrections and guardrails**

- Historical probabilities match saved reference tables in all eight reconstructed groups to ≤ 1.94e−16; this does **not** prove weights are bit-identical to an unavailable original checkpoint.
- Twitter-COVID19 `D` (own threshold) and `D_shared` (C's threshold) differ. The manuscript reports `D_shared` Macro-F1 0.46046, MCC −0.07020.
- SHACL *parsing* and structural checks are not independent SHACL conformance.
- No expert plausibility evaluation or OWL-DL reasoning was carried out.
