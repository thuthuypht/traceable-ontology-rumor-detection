# Consolidated Ontology Explanation / Robustness Audit

**Type:** retrospective evidence audit of already completed runs. No new training or external-test tuning.

## Core comparison (full-tree, original policy)

- **T15→T16**: C=0.7797, D=0.8036, Δ(D−C)=+0.0239 (n=634).
- **T16→T15**: C=0.7773, D=0.7778, Δ(D−C)=+0.0005 (n=1310).

## External COVID-19

- Shared-threshold Macro-F1: C=0.4744, D=0.4605, Δ=-0.0139.
- This previously viewed COVID-19 corpus is now a **development-observed external set**, not a new untouched confirmation for further method revisions.

## Ontology evidence validity

- Sample score-to-RDF matches: 168/168; missing 0; mismatches 0.
- COVID-19 probability-to-RDF matches: 30/30 *sampled predictions*; the other 370 test cases have no sampled RDF.
- These checks establish **technical traceability only**. No annotator review, verification of natural-language claims, rule entailment, or causal attribution was performed.

## Publication posture

1. **Representation:** formalize evidence entities, rules, scores, premises and model decisions with versioned RDF/PROV-O; distinguish semantic rule validity from graph syntactic validity.
2. **Faithfulness:** report rule-perturbation probability and decision-flip rates, and negative/null interventions. Avoid causal interpretations for one-feature replacement.
3. **Robustness:** report cross-policy prediction stability and temporal-window stratification as sensitivity, not select the most favorable test result.
4. **Generalization:** report source/test shift, target poor performance and failures on small trees. A new improved model requires new, untouched fourth dataset.
5. **Human evaluation:** blinded 216-case evaluation is prepared but not rated. Do not claim expert validation unless two independent qualified raters truly score the cases.

## Recommended paper claim

The approach offers an **auditable ontology-guided structural evidence representation** whose predictive and cross-domain limitations are explicitly measured. It has **not** demonstrated consistent accuracy gains attributable to ontology.

## Methodological limits

Results are conditional on existing predictors, source-trained model families, source preprocessing and already observed tests. Input tree-cleaning policies involve refitting and are not pure inference-time perturbations. No formal OWL-DL entailment or expert validated semantics.
