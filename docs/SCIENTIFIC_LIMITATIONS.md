# Limitations that must accompany any research release

1. Rule-score formula reproduction ≠ semantic truth, expert evaluation, or ontology consistency/entailment.
2. Median-substitution effect on CatBoost probability ≠ causal effect of social behavior; interdependent features are not jointly intervened on.
3. Selection of 216 cases was purposive and covers only 135 source threads. Do not treat 1296 interventions as independent statistical replicates.
4. Broad Amplification and several other rules are saturated or redundant with raw graph features.
5. Twitter15/16 held-out directions were repeatedly inspected during development; multiple policy comparisons reduce inferential strength.
6. External Twitter-COVID19 with 400 cases does **not** show matched-threshold Macro-F1 improvement from ontology (C 0.4744 > D_shared 0.4605); the target domain uses a reduced feature space and does not contain the same user identities.
7. RDF structural checks passed for recorded materialization; a real SHACL-engine result is **not** included and must not be described as performed.
8. Funding, author contributions, data licenses, IRB/ethics and code licensing must be resolved by authors before journal submission or public code release.
