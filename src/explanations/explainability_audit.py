"""Reproducible post-hoc ontology explanation/robustness audit.
No training, tuning, target-label model selection, or human expert annotation.
Source inputs are previously completed result files and a prebuilt blinded review form.
"""
from __future__ import annotations
import io, json, math, hashlib, zipfile, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from turtle_lite import Graph, Namespace, RDF, URIRef
# Dependency-free restricted Turtle reader for the archived provenance format.

ROOT = Path(__file__).resolve().parent
EX = Namespace('https://example.org/rumordetection/')
EXTERNAL = Namespace('https://example.org/rumor-validation/')
PROV = Namespace('http://www.w3.org/ns/prov#')


def read_csv(archive, ending, dtype=None):
    name = next((s for s in archive.namelist() if s.endswith(ending)), None)
    if name is None: raise FileNotFoundError(ending)
    return pd.read_csv(io.BytesIO(archive.read(name)), dtype=dtype)


def read_bytes(archive, ending):
    name = next((s for s in archive.namelist() if s.endswith(ending)), None)
    if name is None: raise FileNotFoundError(ending)
    return archive.read(name)


def save_csv(rows, path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(p,index=False)


def direction_label(name): return name.replace('twitter15_to_twitter16','T15→T16').replace('twitter16_to_twitter15','T16→T15')


def main(data_dir, output_dir):
    d=Path(data_dir);o=Path(output_dir);o.mkdir(parents=True,exist_ok=True)
    sens=zipfile.ZipFile(d/'sensitivity_results.zip')
    ext=zipfile.ZipFile(d/'external_results.zip')
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [d/'sensitivity_results.zip',d/'external_results.zip']}
    # Raw predictions from the frozen, historical runs; preserve Snowflake IDs as text.
    p=read_csv(sens,'predictions_by_thread.csv',dtype={'id':str,'dataset':str})
    e=read_csv(ext,'external_predictions.csv',dtype={'id':str})
    rules=read_csv(sens,'ontology/rule_coverage_redundancy.csv')
    fs=read_csv(sens,'explanations/rule_sensitivity_summary.csv')
    erules=read_csv(ext,'ontology_rule_coverage_and_redundancy.csv')
    efid=read_csv(ext,'ontology_model_fidelity.csv')
    # 1. Macrof1 C/D across window, policy, direction. Paired matching uses identical sample IDs.
    rows=[]
    for (policy,window,direction),g in p.groupby(['policy','window','direction'],dropna=False,sort=True):
        c=f1_score(g.label,g.pred_C,average='macro',zero_division=0)
        f=f1_score(g.label,g.pred_D,average='macro',zero_division=0)
        rows.append({'policy':policy,'window':str(window),'direction':direction,'n':len(g),'C_macro_f1':c,'D_macro_f1':f,'delta_D_minus_C':f-c,'disagreement_rate':np.mean(g.pred_C!=g.pred_D),'mean_abs_probability_difference':np.mean(np.abs(g.p_C-g.p_D))})
    save_csv(rows,o/'tables'/'01_ablation.csv')
    # 2. Paired consistency across tree policies, fixed window and direction, common retained IDs only.
    stab=[]
    base=p[p.policy=='original']
    for (policy,window,direction),g in p[p.policy!='original'].groupby(['policy','window','direction']):
        gg=base[(base.window==window)&(base.direction==direction)]
        j=gg[['dataset','id','p_C','p_D','pred_C','pred_D']].merge(g[['dataset','id','p_C','p_D','pred_C','pred_D']],on=['dataset','id'],suffixes=('_base','_new'),validate='one_to_one')
        stab.append({'policy':policy,'window':str(window),'direction':direction,'n_shared':len(j),'n_original':len(gg),'n_new':len(g),
                     'D_probability_drift_mean_abs':np.abs(j.p_D_base-j.p_D_new).mean(),
                     'D_prediction_flip_rate':np.mean(j.pred_D_base!=j.pred_D_new),
                     'C_prediction_flip_rate':np.mean(j.pred_C_base!=j.pred_C_new),
                     'delta_disagreement_rate':np.mean(j.pred_C_new!=j.pred_D_new)-np.mean(j.pred_C_base!=j.pred_D_base)})
    save_csv(stab,o/'tables'/'02_policy_prediction_stability.csv')
    # 3. Separate rule informativeness from structural serialization.
    rule_stats=rules[ (rules.policy=='original') & (rules.window.astype(str)=='full') ].copy()
    for col in ['saturation_ge_095','near_zero_le_005','max_abs_pearson']:
        rule_stats[col]=pd.to_numeric(rule_stats[col],errors='coerce')
    rule_stats['flag_saturated_95pct']=rule_stats.saturation_ge_095>=0.95
    rule_stats['flag_redundant_abs_r_ge_095']=rule_stats.max_abs_pearson>=0.95
    rule_stats.to_csv(o/'tables'/'03_rules_saturation_redundancy.csv',index=False)
    fs.to_csv(o/'tables'/'04_rule_model_sensitivity_T1516.csv',index=False)
    erules.merge(efid,on='rule',how='outer').to_csv(o/'tables'/'05_rule_model_sensitivity_COVID19.csv',index=False)
    # 4. Machine-checkable provenance (does not assert semantically correct justifications).
    gs=Graph();gs.parse(data=read_bytes(sens,'ontology_provenance_sample.ttl').decode('utf-8'),format='turtle')
    ge=Graph();ge.parse(data=read_bytes(ext,'ontology_provenance.ttl').decode('utf-8'),format='turtle')
    sample=read_csv(sens,'ontology_provenance_sample.csv',dtype={'id':str})
    mismatches=[];matched=0;missing=0
    for _,r in sample.iterrows():
        # Use fixed IRI convention from the prior pipeline, not NLP matching.
        uri=EX[f"act_{r['dataset']}_{r['id']}_o_{str(r['rule']).strip()}"]
        # Existing TTL IRIs contain snake-case feature names; identify by rule URI instead.
        obs=URIRef(str(EX[f"thread_{r['dataset']}_{r['id']}"]))
        candidates=[s for s in gs.subjects(PROV.wasDerivedFrom,obs) if str(s).split('/')[-1].startswith(f"act_{r['dataset']}_{r['id']}_")]
        target=[s for s in candidates if str(gs.value(s,EX.rule) or '').split('/')[-1]==str(r['rule'])]
        if len(target)!=1:missing+=1;continue
        q=gs.value(target[0],EX.confidence)
        if q is None:missing+=1;continue
        matched+=1
        if abs(float(q)-float(r.score))>1.1e-6:mismatches.append({'dataset':r.dataset,'id':r.id,'rule':r.rule,'from_csv':r.score,'from_ttl':float(q)})
    save_csv(mismatches,o/'tables'/'06_provenance_score_mismatches.csv')
    epreds=list(ge.subjects(RDF.type,EXTERNAL.ExternalPrediction))
    with_evidence=sum(len(list(ge.objects(s,EXTERNAL.hasEvidence)))>=1 for s in epreds)
    ext_prob_matches=0;ext_prob_missing=0;ext_prob_mismatch=0
    for s in epreds:
        try:ix=int(str(s).rsplit('/',1)[-1])
        except ValueError:ext_prob_missing+=1;continue
        if ix<0 or ix>=len(e):ext_prob_missing+=1;continue
        q=ge.value(s,EXTERNAL.predictedRumorProbability)
        if q is None:ext_prob_missing+=1
        elif abs(float(q)-float(e.iloc[ix].p_D))<=2e-6:ext_prob_matches+=1
        else:ext_prob_mismatch+=1
    prov_report={'rdf_validation_mode':'restricted_turtle_snapshot_parser_not_full_rdf_validation','sensitivity_rdf_triples':len(gs),'sensitivity_sample_scores':len(sample),'sensitivity_score_matches_found':matched,'sensitivity_scores_missing':missing,'sensitivity_numeric_mismatches':len(mismatches),
                 'covid_rdf_triples':len(ge),'covid_prediction_nodes':len(epreds),'covid_prediction_nodes_with_evidence':with_evidence,
                 'covid_probability_matches':ext_prob_matches,'covid_sample_probability_missing':ext_prob_missing,'covid_probability_mismatch':ext_prob_mismatch,'covid_predictions_without_sampled_rdf':len(e)-len(epreds),
                 'scope':'Serialization, referenced entities, and numeric consistency only: NO expert verification, veracity adjudication, causal evidence, or OWL-DL proof.'}
    (o/'reports'/'provenance_validation.json').parent.mkdir(exist_ok=True)
    (o/'reports'/'provenance_validation.json').write_text(json.dumps(prov_report,indent=2),encoding='utf-8')
    # 5. External zero-shot analysis without retuning thresholds.
    evals=[]
    shared_threshold_stats={}
    for m in ['A','B','C','D']:
        y=e.label.to_numpy(int);prob=e['p_'+m].to_numpy(float);pred=e['pred_'+m].to_numpy(int)
        evals.append({'model':m,'n':len(y),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
                      'auprc':average_precision_score(y,prob),'roc_auc':roc_auc_score(y,prob),'mcc':matthews_corrcoef(y,pred),
                      'positive_predictions':int(pred.sum())})
    for m in ['C','D']:
        pred=e[f'pred_{m}_shared'].to_numpy(int)
        shared_threshold_stats[m]=f1_score(e.label,pred,average='macro',zero_division=0)
    save_csv(evals,o/'tables'/'07_external_performance.csv')
    # D-C is same-threshold comparison; bootstrap is a DESCRIPTIVE test-set uncertainty interval.
    ci=read_csv(ext,'paired_bootstrap.csv')
    ci.to_csv(o/'tables'/'08_external_paired_bootstrap.csv',index=False)
    # Model stability by propagation size. This is NOT a new model or a model-selection step.
    tree=read_csv(ext,'target_tree_integrity.csv',dtype={'id':str})
    merged=e.merge(tree[['id','valid_nodes']],on='id',how='inner',validate='one_to_one')
    merged['size_quartile']=pd.qcut(merged.valid_nodes.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
    by_size=[]
    for q,g in merged.groupby('size_quartile',observed=True):
        by_size.append({'size_quartile':str(q),'n':len(g),'min_nodes':g.valid_nodes.min(),'max_nodes':g.valid_nodes.max(),
                        'label_rumor_rate':g.label.mean(),'pred_C_rumor_rate':g.pred_C_shared.mean(),'pred_D_rumor_rate':g.pred_D_shared.mean(),
                        'C_macro_f1':f1_score(g.label,g.pred_C_shared,average='macro',zero_division=0),
                        'D_macro_f1':f1_score(g.label,g.pred_D_shared,average='macro',zero_division=0),
                        'p_C_mean':g.p_C.mean(),'p_D_mean':g.p_D.mean()})
    save_csv(by_size,o/'tables'/'09_domain_shift_by_tree_size.csv')
    # Source OOF scores are not comparable estimates of same target distribution.
    source=read_csv(ext,'source_only_oof_model_selection.csv')
    source.to_csv(o/'tables'/'10_source_only_selection_reference.csv',index=False)
    # 6. Human review: flag blank ratings, calculate NOTHING as expert adjudication.
    reviewer=d/'ANNOTATOR_RESPONSES_EMPTY.csv'
    if reviewer.exists():
        annotations=pd.read_csv(reviewer,dtype=str).fillna('')
        dimensions=['evidence_available','premise_supported','rule_semantically_valid','temporally_valid']
        present=0
        for dim in dimensions:
            if dim in annotations.columns:present+=int((annotations[dim].str.strip()!='').sum())
        human={'rows':len(annotations),'filled_dimension_scores':present,'independent_expert_validation_completed':False,'advice':'Obtain real independent ratings and credentials; never impute or simulate expert agreement.'}
    else:human={'rows':0,'filled_dimension_scores':0,'independent_expert_validation_completed':False,'advice':'Reviewer form unavailable.'}
    (o/'reports'/'human_annotation_status.json').write_text(json.dumps(human,indent=2),encoding='utf-8')
    # 7. Paper-ready note, no post hoc model selection.
    original=[x for x in rows if x['policy']=='original' and x['window']=='full']
    main_by_direction={direction_label(x['direction']):x for x in original}
    report='''# Consolidated Ontology Explanation / Robustness Audit\n\n'''
    report+='**Type:** retrospective evidence audit of already completed runs. No new training or external-test tuning.\n\n'
    report+='## Core comparison (full-tree, original policy)\n\n'
    for name,rx in main_by_direction.items():report+=f"- **{name}**: C={rx['C_macro_f1']:.4f}, D={rx['D_macro_f1']:.4f}, Δ(D−C)={rx['delta_D_minus_C']:+.4f} (n={rx['n']}).\n"
    report+='\n## External COVID-19\n\n'
    report+=f"- Shared-threshold Macro-F1: C={shared_threshold_stats['C']:.4f}, D={shared_threshold_stats['D']:.4f}, Δ={shared_threshold_stats['D']-shared_threshold_stats['C']:+.4f}.\n"
    report+='- This previously viewed COVID-19 corpus is now a **development-observed external set**, not a new untouched confirmation for further method revisions.\n'
    report+='\n## Ontology evidence validity\n\n'
    report+=f"- Sample score-to-RDF matches: {matched}/{len(sample)}; missing {missing}; mismatches {len(mismatches)}.\n"
    report+=f"- COVID-19 probability-to-RDF matches: {ext_prob_matches}/{len(epreds)} *sampled predictions*; the other {len(e)-len(epreds)} test cases have no sampled RDF.\n"
    report+='- These checks establish **technical traceability only**. No annotator review, verification of natural-language claims, rule entailment, or causal attribution was performed.\n'
    report+='\n## Publication posture\n\n'
    report+='1. **Representation:** formalize evidence entities, rules, scores, premises and model decisions with versioned RDF/PROV-O; distinguish semantic rule validity from graph syntactic validity.\n'
    report+='2. **Faithfulness:** report rule-perturbation probability and decision-flip rates, and negative/null interventions. Avoid causal interpretations for one-feature replacement.\n'
    report+='3. **Robustness:** report cross-policy prediction stability and temporal-window stratification as sensitivity, not select the most favorable test result.\n'
    report+='4. **Generalization:** report source/test shift, target poor performance and failures on small trees. A new improved model requires new, untouched fourth dataset.\n'
    report+='5. **Human evaluation:** blinded 216-case evaluation is prepared but not rated. Do not claim expert validation unless two independent qualified raters truly score the cases.\n'
    report+='\n## Recommended paper claim\n\n'
    report+='The approach offers an **auditable ontology-guided structural evidence representation** whose predictive and cross-domain limitations are explicitly measured. It has **not** demonstrated consistent accuracy gains attributable to ontology.\n'
    report+='\n## Methodological limits\n\n'
    report+='Results are conditional on existing predictors, source-trained model families, source preprocessing and already observed tests. Input tree-cleaning policies involve refitting and are not pure inference-time perturbations. No formal OWL-DL entailment or expert validated semantics.\n'
    (o/'reports'/'PAPER_EVIDENCE_SUMMARY_EN.md').write_text(report,encoding='utf-8')
    # Source checksums: use hash of uploaded run products, not claim source code hash.
    manifest={'frozen_result_archives_sha256':fingerprints,'audit_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'test_labels_used_for_model_tuning':False,'new_models_trained':False,'ontology_expert_validated':False,'tables':sorted(str(x.relative_to(o)) for x in o.rglob('*.csv'))}
    (o/'reports'/'AUDIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # pack outputs
    path=o.parent/(o.name+'_Results.zip')
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as zz:
        for f in sorted(o.rglob('*')):
            if f.is_file():zz.write(f,f.relative_to(o))
    print('AUDIT COMPLETE',path)
    print(json.dumps({'provenance':prov_report,'external_C_D':shared_threshold_stats,'human_review':human,'frozen_archives':fingerprints},indent=2))
    return path

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data-dir',default=str(ROOT/'embedded_inputs'));a.add_argument('--output-dir',default=str(ROOT/'audit_results'));v=a.parse_args()
    main(v.data_dir,v.output_dir)
