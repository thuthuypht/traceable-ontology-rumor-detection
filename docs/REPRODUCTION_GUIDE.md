# Reproduction guide / provenance boundaries

**Quick check (no data, CPU, offline):** from repo root run `python scripts/verify_paper_tables.py`, then optionally run `notebooks/Verify_Paper_Tables_Offline.ipynb`.

**Full reconstruction (requires authorized data access):**
1. Create a virtual environment using `requirements.txt` and record exact versions; original model manifest used Python 3.13.18, scikit-learn 1.6.1, CatBoost 1.2.10.
2. Obtain Twitter15/16 source posts and propagation trees from pinned GitHub commits, and Twitter-COVID19 from its original research archive (URL and SHA-256 recorded in frozen protocol).
3. Use `src/twitter/frozen_twitter1516_pipeline.py` for features and source-only selection; `src/twitter/sensitivity_ontology_audit.py` for tree policy checks; `src/external/external_validation_pipeline.py` for target-label-blind COVID19 transfer. These scripts have their original CLI flags; inspect `--help` or source before running. External script imports sibling dependencies; keep its dependency scripts together on PYTHONPATH or in one working directory.
4. Run `src/explanations/ontology216_auto_audit.py` against the 216-case audit and source trees.
5. For the full CatBoost artifact reconstruction, assemble `sources.csv`, `cases_216.csv`, `reference_predictions.csv`, `original.csv`, `strict.csv` from the separate authorized study archive and run `src/explanations/full_provenance_faithfulness.py --input-dir INPUT --output-dir OUTPUT`.
6. Verify the produced models and scores: `src/explanations/verify_release.py --root OUTPUT` (inspect its CLI first), and optionally run SHACL via `scripts/run_shacl_validation.py` on your full RDF file.
7. Confirm that your outputs match published tables and check SHA-256 manifests. **Do not optimize on the already-used Twitter-COVID19 target labels** and present it as new holdout validation.

**Important:** `notebooks/Verify_Paper_Tables_Offline.ipynb` is a numerical report check, not a training/reconstruction notebook. The original Run-All notebooks used embedded research data and are available only in the **private** area of the downloadable package, for authorized users.
