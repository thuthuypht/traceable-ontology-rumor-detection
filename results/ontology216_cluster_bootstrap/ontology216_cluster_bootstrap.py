"""Cluster-resampled 216-case Ontology sensitivity statistics.

Study estimand: among the 135 purposely selected source threads, each thread
has equal weight after averaging all observed case configurations from it.
Bootstrap scheme: 10,000 stratified source-thread cluster draws (70 Twitter15,
65 Twitter16) with replacement, preserving every case and six-rule comparison
inside each selected thread. No model training or respecification.

Importantly: these percentile intervals do NOT establish external-population
coverage, semantic truth, causal validity or superiority of ontology rules.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

RULES = (
    "o_rapid_independent_burst",
    "o_deep_persistence",
    "o_broad_amplification",
    "o_delayed_revival",
    "o_multi_branch_diffusion",
    "o_temporal_concentration",
)
METRICS = ("mean_abs_delta_p", "fraction_flipped", "fraction_abs_delta_ge_001")
EXPECTED_ROWS, EXPECTED_CASES, EXPECTED_THREADS = 1296, 216, 135


def assert_valid(df: pd.DataFrame):
    required = {"case_id", "dataset", "thread_key", "direction", "policy", "window", "rule", "abs_delta_p", "delta_p", "flipped_label"}
    miss = required.difference(df.columns)
    if miss:
        raise ValueError("Missing columns: " + ", ".join(sorted(miss)))
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows; found {len(df)}")
    if df.case_id.nunique() != EXPECTED_CASES or df.thread_key.nunique() != EXPECTED_THREADS:
        raise ValueError("Expected 216 cases from 135 independent source-thread clusters")
    if df.duplicated(["case_id", "rule"]).any():
        raise ValueError("Duplicate case–rule pairs")
    if set(df.rule) != set(RULES):
        raise ValueError("Unexpected Ontology rule names")
    if not (df.groupby("case_id").rule.nunique() == 6).all():
        raise ValueError("Not all cases have exactly six rule interventions")
    for fld in ("thread_key", "dataset", "direction", "policy", "window"):
        if (df.groupby("case_id")[fld].nunique(dropna=False) > 1).any():
            raise ValueError(f"Case has contradictory {fld}")
    if (df.groupby("thread_key").dataset.nunique() > 1).any():
        raise ValueError("Thread key spans datasets: use dataset + source ID grouping")
    if not set(df.dataset) == {"twitter15", "twitter16"}:
        raise ValueError("Expected twitter15 and twitter16 datasets")
    for fld in ("abs_delta_p", "delta_p", "flipped_label"):
        if not np.isfinite(pd.to_numeric(df[fld])).all():
            raise ValueError(f"Nonfinite metric: {fld}")
    if not df.flipped_label.isin([0, 1]).all():
        raise ValueError("flipped_label must be 0 or 1")
    if not np.allclose(df.abs_delta_p.to_numpy(), np.abs(df.delta_p.to_numpy()), atol=1e-12, rtol=0):
        raise ValueError("abs_delta_p and |delta_p| do not agree")
    sizes = df.groupby("case_id").size()
    if not (sizes == 6).all():
        raise ValueError("Each case must contribute exactly six observations")
    return {"rows": int(len(df)), "cases": int(df.case_id.nunique()), "source_threads": int(df.thread_key.nunique()),
            "cases_per_source_thread_histogram": {str(k):int(v) for k,v in df.groupby('thread_key').case_id.nunique().value_counts().sort_index().items()},
            "threads_per_dataset": {k:int(v) for k,v in df.groupby('dataset').thread_key.nunique().items()},
            "flips_total": int(df.flipped_label.sum()),
            "selected_source_threads_with_any_flip": int(df.groupby('thread_key').flipped_label.max().sum())}


def analyze(input_csv: str|Path, output_dir: str|Path, *, B: int = 10000, seed: int = 42) -> dict:
    input_csv = Path(input_csv); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, dtype={"case_id":str,"dataset":str,"thread_key":str,"direction":str,"policy":str,"window":str,"rule":str})
    validation = assert_valid(df)
    df = df.copy()
    df["flipped_label"] = df["flipped_label"].astype(int)
    df["abs_ge_001"] = (df["abs_delta_p"] >= 0.01).astype(int)
    groups = sorted(df.thread_key.unique())
    group_dataset = df.drop_duplicates('thread_key').set_index('thread_key').loc[groups,'dataset']
    per_thread_casecounts = df.groupby('thread_key').case_id.nunique().reindex(groups).to_numpy(dtype=float)
    G, R, M = len(groups), len(RULES), len(METRICS)
    thread_avg = np.zeros((G,R,M)); thread_sum = np.zeros((G,R,M))
    for r, rule in enumerate(RULES):
        d = df[df.rule == rule]
        cols = ('abs_delta_p','flipped_label','abs_ge_001')
        thread_avg[:,r,:] = d.groupby('thread_key')[list(cols)].mean().reindex(groups).to_numpy()
        thread_sum[:,r,:] = d.groupby('thread_key')[list(cols)].sum().reindex(groups).to_numpy()
    if not np.isfinite(thread_avg).all():
        raise ValueError("Incomplete rule-by-thread matrix")
    if not (thread_avg[:,:,1] >= 0).all():
        raise ValueError('Invalid flip indicator')
    idx_by_dataset = {dataset: np.flatnonzero((group_dataset.to_numpy()==dataset)) for dataset in ('twitter15','twitter16')}
    assert sorted(map(len,idx_by_dataset.values()))==[65,70]
    point_thread = thread_avg.mean(axis=0)
    point_case = thread_sum.sum(axis=0)/per_thread_casecounts.sum()
    boot_thread=np.empty((B,R,M),float); boot_case=np.empty((B,R,M),float)
    rng = np.random.default_rng(seed)
    for start in range(0,B,400):
        end=min(B,start+400); batch=end-start
        sampleidx=np.concatenate([rng.choice(ids,size=(batch,len(ids)),replace=True)
                                  for ids in idx_by_dataset.values()],axis=1)
        # All six rules and all repeated case configurations remain together within each source thread.
        boot_thread[start:end]=thread_avg[sampleidx].mean(axis=1)
        boot_case[start:end]=thread_sum[sampleidx].sum(axis=1)/per_thread_casecounts[sampleidx].sum(axis=1)[:,None,None]
    reports=[]
    for name,observed,boots in [('thread_equal',point_thread,boot_thread),('case_weighted',point_case,boot_case)]:
        for j,rule in enumerate(RULES):
            record={'rule':rule,'estimand':name,'n_case_rule_pairs':EXPECTED_CASES,'n_source_threads':G,'bootstrap_reps':B}
            for m,metric in enumerate(METRICS):
                dist=boots[:,j,m]
                record[metric]=float(observed[j,m]);record[metric+'_ci_low']=float(np.quantile(dist,.025));record[metric+'_ci_high']=float(np.quantile(dist,.975));record[metric+'_bootstrap_se']=float(dist.std(ddof=1))
            reports.append(record)
    combined=pd.DataFrame(reports)
    combined[combined.estimand=='thread_equal'].to_csv(out/'Table_ThreadEqual_ClusterBootstrap_216.csv',index=False,float_format='%.10g')
    combined[combined.estimand=='case_weighted'].to_csv(out/'Table_CaseWeighted_ClusterBootstrap_216.csv',index=False,float_format='%.10g')
    contrast=(thread_avg[:,0,0]-thread_avg[:,2,0]).mean()
    dist=boot_thread[:,0,0]-boot_thread[:,2,0]
    pd.DataFrame([{'contrast':'rapid_independent_burst minus broad_amplification',
                   'estimand':'thread_equal','exploratory':True,'n_source_threads':G,
                   'difference_mean_abs_delta_p':float(contrast),
                   'ci_low':float(np.quantile(dist,.025)),'ci_high':float(np.quantile(dist,.975))
                 }]).to_csv(out/'Exploratory_RuleContrast.csv',index=False)
    thread_sizes = df.groupby(['dataset','thread_key'],as_index=False).agg(n_case_ids=('case_id','nunique'),n_rule_interventions=('rule','size'),any_decision_flip=('flipped_label','max'))
    thread_sizes.to_csv(out/'Thread_Cluster_Size_Audit.csv',index=False)
    by_config = df.groupby(['direction','policy','window','rule'],as_index=False).agg(n_cases=('case_id','nunique'),n_threads=('thread_key','nunique'),mean_abs_delta_p=('abs_delta_p','mean'),flip_fraction=('flipped_label','mean'))
    by_config.to_csv(out/'Supplemental_Config_Descriptives.csv',index=False)
    np.savez_compressed(out/'Bootstrap_Replicates_10k.npz',thread_equal=boot_thread,case_weighted=boot_case,
                        rule_order=np.asarray(RULES),metric_order=np.asarray(METRICS))
    validation.update({'bootstrap_repetitions':B,'seed':seed,
        'primary_estimand':'equal weight to each observed source thread after averaging its observed configurations; stratified resampling within Twitter15/Twitter16',
        'secondary_estimand':'case-weighted mean across the purposively selected 216 case records, CI using source-thread resampling',
        'ci_type':'two-sided percentile cluster bootstrap (2.5, 97.5)',
        'bootstrap_notes':'conditional sensitivity analysis on a purposively selected sample: cannot claim population coverage, novel external generalization or causal inference',
        'input_file':input_csv.name,
        'input_sha256':hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        'numpy':np.__version__,'pandas':pd.__version__, 'python':platform.python_version()})
    (out/'Analysis_Manifest.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False),encoding='utf-8')
    pivot=combined[combined.estimand=='thread_equal'].set_index('rule')
    pivotcase=combined[combined.estimand=='case_weighted'].set_index('rule')
    lines=[
      '# Cluster-aware uncertainty analysis (Ontology 216)', '',
      f'Statistical unit: {G} source threads, not 216 independent records; 70 Twitter15 and 65 Twitter16 threads.',
      f'{B:,} stratified thread-level bootstrap replicates, seed {seed}. All per-thread cases and six interventions were kept intact.',
      '', '## Table for the IEEE manuscript (thread-equal estimand)', '',
      '| Rule | Mean |Δp| | 95% cluster-bootstrap percentile interval | Flip fraction | 95% interval |',
      '|---|---:|---:|---:|---:|'
    ]
    short={'o_rapid_independent_burst':'Rapid Independent Burst','o_deep_persistence':'Deep Persistence',
      'o_broad_amplification':'Broad Amplification','o_delayed_revival':'Delayed Revival',
      'o_multi_branch_diffusion':'Multi-Branch Diffusion','o_temporal_concentration':'Temporal Concentration'}
    for rule in RULES:
        x=pivot.loc[rule]
        lines.append(f'| {short[rule]} | {x.mean_abs_delta_p:.4f} | [{x.mean_abs_delta_p_ci_low:.4f}, {x.mean_abs_delta_p_ci_high:.4f}] | {x.fraction_flipped:.3%} | [{x.fraction_flipped_ci_low:.3%}, {x.fraction_flipped_ci_high:.3%}] |')
    lines+=['','## How this differs from the existing Table VII','',
            'The earlier Table VII reports **case-weighted** mean |Δp|: all 216 case records receive equal weight. The above primary table weights each of the 135 distinct threads equally; repeated case windows/policies within a thread are averaged before bootstrap.',
            'A secondary table retains the original case-weighted point estimates, but its uncertainty is also clustered by source thread.',
            '', '## Reporting limits','',
            'The 216 records are purposively selected, include 48 C–D disagreement cases, and come from 135 threads. Bootstrap intervals quantify instability under resampling of these observed threads, not unbiased confidence bounds for all social-media threads. No claim about semantic correctness or causal effects follows.',
            '', '## Suggested additional reference (real, verified)','',
            'C. A. Field and A. H. Welsh, “Bootstrapping clustered data,” *J. R. Stat. Soc. Ser. B*, vol. 69, no. 3, pp. 369–390, 2007, doi: 10.1111/j.1467-9868.2007.00593.x.',
            '']
    (out/'IEEE_Table_and_Interpretation_EN.md').write_text('\n'.join(lines),encoding='utf-8')
    return validation

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True);ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args();v=analyze(a.input,a.output,B=a.bootstrap,seed=a.seed)
    print(json.dumps({k:v[k] for k in ('cases','source_threads','threads_per_dataset','flips_total','bootstrap_repetitions')},indent=2))
