# Automated Ontology Explanation Verification — 216 case audit

**This is NOT expert validation.** It audits deterministic rule execution and available provenance only.

- Cases: **216**. Original trees available for 216 cases.
- Scores verified against original trees: **216**.
- Scores recomputed from visible rounded evidence: **164**.
- Cases with insufficient *visible* evidence: **52**.
- Verified mismatches: **0**.

## Interpret scores correctly

- **2**: rule score and visible evidence agree with original propagation tree.
- **1**: score rederived from rounded displayed features only; source tree not verified.
- **0**: a demonstrable score/evidence/provenance mismatch.
- **blank**: insufficient evidence, NOT a failed rule.

These are **computational reproducibility levels**, not human ratings of semantic validity or usefulness.
No verification of rumor truth, support/deny stance, semantic explanation quality, OWL-DL entailment, or causal faithfulness is claimed.

## Six rules

| Rule | Cases | Raw pass | Visible pass | Visible incomplete | Saturated >=.95 |
|---|---:|---:|---:|---:|---:|
| o_rapid_independent_burst | 36 | 36 | 28 | 8 | 25 |
| o_deep_persistence | 36 | 36 | 0 | 36 | 5 |
| o_broad_amplification | 36 | 36 | 36 | 0 | 30 |
| o_delayed_revival | 36 | 36 | 36 | 0 | 0 |
| o_multi_branch_diffusion | 36 | 36 | 28 | 8 | 10 |
| o_temporal_concentration | 36 | 36 | 36 | 0 | 0 |

## Interpreting model faithfulness

Absolute p_D−p_C is a **model-comparison** diagnostic, not the causal contribution of the listed single rule.
Any group-level rule sensitivity is from a prior median-substitution analysis of the trained model, not a case-specific counterfactual.
To measure individual-rule intervention faithfully, load the frozen trained model and perturb the original rule feature (and ideally its correlated evidence).

## Source provenance

- Original repository SHA: `6a33ebd91d4cafeb5a42eabc5727226fec7bb0e5`
- Raw trees: `PINNED_GITHUB_ZIP`
- Automated numeric tolerance: `0.005` (features shown at four decimals).
