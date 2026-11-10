# Cluster-aware uncertainty analysis (Ontology 216)

Statistical unit: 135 source threads, not 216 independent records; 70 Twitter15 and 65 Twitter16 threads.
10,000 stratified thread-level bootstrap replicates, seed 42. All per-thread cases and six interventions were kept intact.

## Table for the IEEE manuscript (thread-equal estimand)

| Rule | Mean |Δp| | 95% cluster-bootstrap percentile interval | Flip fraction | 95% interval |
|---|---:|---:|---:|---:|
| Rapid Independent Burst | 0.0162 | [0.0135, 0.0191] | 5.852% | [2.444%, 10.000%] |
| Deep Persistence | 0.0091 | [0.0066, 0.0119] | 4.370% | [1.481%, 7.778%] |
| Broad Amplification | 0.0006 | [0.0000, 0.0014] | 0.000% | [0.000%, 0.000%] |
| Delayed Revival | 0.0086 | [0.0064, 0.0112] | 2.370% | [0.148%, 5.185%] |
| Multi-Branch Diffusion | 0.0028 | [0.0019, 0.0039] | 2.222% | [0.000%, 5.185%] |
| Temporal Concentration | 0.0058 | [0.0041, 0.0076] | 2.963% | [0.741%, 5.926%] |

## How this differs from the existing Table VII

The earlier Table VII reports **case-weighted** mean |Δp|: all 216 case records receive equal weight. The above primary table weights each of the 135 distinct threads equally; repeated case windows/policies within a thread are averaged before bootstrap.
A secondary table retains the original case-weighted point estimates, but its uncertainty is also clustered by source thread.

## Reporting limits

The 216 records are purposively selected, include 48 C–D disagreement cases, and come from 135 threads. Bootstrap intervals quantify instability under resampling of these observed threads, not unbiased confidence bounds for all social-media threads. No claim about semantic correctness or causal effects follows.

## Suggested additional reference (real, verified)

C. A. Field and A. H. Welsh, “Bootstrapping clustered data,” *J. R. Stat. Soc. Ser. B*, vol. 69, no. 3, pp. 369–390, 2007, doi: 10.1111/j.1467-9868.2007.00593.x.
