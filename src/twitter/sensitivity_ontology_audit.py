"""One-run Twitter15/Twitter16 sensitivity and structural-fuzzy-ontology evidence audit.

Scientific protocol: frozen label-blind policies; same previously source-OOF C threshold
for D, C, all cleaning policies; target labels exclusively for reporting. Not OWL-DL reasoning.
Source Git snapshots and binary rumor mapping are defined in frozen_twitter1516_pipeline.py.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, time, zipfile
from collections import defaultdict, Counter, deque
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import rdflib
from rdflib import RDF, URIRef, Literal, Namespace
from rdflib.namespace import XSD
import frozen_twitter1516_pipeline as ref

POLICIES=['original','exclude_negative_trees','strict_reachable_tree']
WINDOWS=['60','360','1440','full']
DIRECTIONS=[('twitter15','twitter16'),('twitter16','twitter15')]
# Source-only grouped OOF thresholds determined in the previous frozen run.
# These are *not* selected using target labels, and are identical for C,D, all policies.
FROZEN_C_THRESHOLDS={
 ('twitter15_to_twitter16','60'):.625,('twitter16_to_twitter15','60'):.550,
 ('twitter15_to_twitter16','360'):.500,('twitter16_to_twitter15','360'):.525,
 ('twitter15_to_twitter16','1440'):.475,('twitter16_to_twitter15','1440'):.575,
 ('twitter15_to_twitter16','full'):.475,('twitter16_to_twitter15','full'):.650,
}
G=ref.G_COLS; E=ref.R_COLS; O=ref.O_COLS
SEED=42

def save_csv(out,name,rows,columns=None):
    p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=columns).to_csv(p,index=False)

def strict_tree(old):
    """One-parent reachable directed temporal arborescence, without negative/backward/self edges.
    Deterministic; preserves (user,tweet,time) identity and synthetic ROOT. Not inferred reply text.
    """
    valid=sorted([(p,c) for p,c in old['edges'] if p[2]>=0 and c[2]>=0 and c[2]+1e-9>=p[2] and p!=c],
                 key=lambda a:(a[0][2],a[1][2],a[0][0],a[1][0],a[1][1]))
    child=defaultdict(list)
    for p,c in valid:child[p].append(c)
    roots=[n for n in old['nodes'] if n[1]=='ROOT']
    if not roots:
        roots=[('ROOT','ROOT',0.)]
    visited=set(roots); queue=deque(sorted(roots));keep=set();
    while queue:
        p=queue.popleft()
        for c in sorted(child.get(p,[]),key=lambda x:(x[2],x[0],x[1])):
            if c not in visited:
                visited.add(c);keep.add((p,c));queue.append(c)
    # No invented edges; disconnected evidence is excluded instead of reattached.
    return {'nodes':visited,'edges':keep,'issues':old['issues'],'examples':old['examples']}

def materialize_features(frame,base,out,windows):
    """Parse exactly once per tree; produce matched original and strict policies."""
    original=[];strict=[];qa=[]
    for idx,r in enumerate(frame.itertuples(index=False),1):
        tree=ref.parse_tree(base/r.dataset/'tree'/f'{r.id}.txt')
        clean=strict_tree(tree)
        qa.append({'dataset':r.dataset,'id':r.id,
            **{f'issue_{k}':int(tree['issues'].get(k,0)) for k in ['duplicate_edges','self_edges','negative_delay','backwards_edge','malformed_lines']},
            'original_nodes':len(tree['nodes']),'original_edges':len(tree['edges']),
            'strict_nodes':len(clean['nodes']),'strict_edges':len(clean['edges']),
            'removed_nodes':len(tree['nodes'])-len(clean['nodes']),
            'removed_edges':len(tree['edges'])-len(clean['edges'])})
        for w in windows:
            a,sa=ref.snapshot_features(tree,w)
            b,sb=ref.snapshot_features(clean,w)
            common={'dataset':r.dataset,'id':r.id,'window':str(w)}
            original.append({**common,**a,**{f'visible_{k}':v for k,v in sa.items()}})
            strict.append({**common,**b,**{f'visible_{k}':v for k,v in sb.items()}})
        if idx%500==0:print('Parsed trees',idx,'/',len(frame),flush=True)
    orig=pd.DataFrame(original);st=pd.DataFrame(strict);issues=pd.DataFrame(qa)
    for d in [orig,st]:
        assert len(d)==len(frame)*len(windows) and np.isfinite(d[G+E+O].to_numpy(dtype=float)).all()
        assert (d[O].to_numpy()>=0).all() and (d[O].to_numpy()<=1).all()
    save_csv(out,'audit/tree_policy_changes.csv',qa)
    summarize=issues.groupby('dataset').agg(threads=('id','count'),negative_trees=('issue_negative_delay',lambda s:int((s>0).sum())),
        backwards_trees=('issue_backwards_edge',lambda s:int((s>0).sum())),duplicate_trees=('issue_duplicate_edges',lambda s:int((s>0).sum())),
        self_edge_trees=('issue_self_edges',lambda s:int((s>0).sum())),strict_nodes_removed=('removed_nodes','sum'),strict_edges_removed=('removed_edges','sum')).reset_index()
    summarize.to_csv(out/'audit/tree_policy_summary.csv',index=False)
    orig.to_csv(out/'features_original.csv',index=False)
    st.to_csv(out/'features_strict.csv',index=False)
    return {'original':orig,'strict_reachable_tree':st,'exclude_negative_trees':orig},issues

def fit_model(train,valid,xt_train,xt_test,cols,seed=42,placebo_seed=None,return_matrix=False):
    xtr=np.hstack([xt_train,ref.features_for(train,cols)])
    xte=np.hstack([xt_test,ref.features_for(valid,cols)])
    if placebo_seed is not None:
        rng=np.random.default_rng(placebo_seed)
        start=xtr.shape[1]-len(O)
        for X,F in [(xtr,train),(xte,valid)]:
            # Permute full ontology-score vectors within tree-size strata independently
            # for source and target; retain marginal distribution and local graph size.
            bins=np.floor(F.g_log_nodes.to_numpy(dtype=float)/np.log(2)).astype(int)
            for b in np.unique(bins):
                inds=np.where(bins==b)[0]
                if len(inds)>1:
                    old=X[inds,start:].copy()
                    X[inds,start:]=old[rng.permutation(len(inds))]
    model=ref.make_predictor(seed)
    model.fit(xtr,train.label.to_numpy(dtype=int))
    p=model.predict_proba(xte)[:,1]
    return (p,model,xtr,xte) if return_matrix else p

def bootstrap_diff(y,pD,pC,thr,repeats,seed):
    rng=np.random.default_rng(seed)
    a=np.asarray(pD)>=thr;b=np.asarray(pC)>=thr;y=np.asarray(y)
    deltas=[]
    for _ in range(repeats):
        ix=rng.integers(0,len(y),len(y))
        deltas.append(f1_score(y[ix],a[ix],average='macro',zero_division=0)-f1_score(y[ix],b[ix],average='macro',zero_division=0))
    return np.quantile(deltas,[.025,.975]).tolist()

def build_rule_audit(features_by_policy,out):
    rows=[];pairs=[]
    for policy,df in features_by_policy.items():
      if policy=='exclude_negative_trees':continue
      for (ds,w),f in df.groupby(['dataset','window'],sort=False):
        for rule in O:
            vals=f[rule].to_numpy(dtype=float)
            rv={'policy':policy,'dataset':ds,'window':w,'rule':rule,'n':len(vals),
                'mean':float(np.mean(vals)),'std':float(np.std(vals)),'min':float(np.min(vals)),
                'p05':float(np.quantile(vals,.05)),'median':float(np.median(vals)),
                'p95':float(np.quantile(vals,.95)),'max':float(np.max(vals)),
                'activation_ge_05':float(np.mean(vals>=.5)),
                'saturation_ge_095':float(np.mean(vals>=.95)),
                'near_zero_le_005':float(np.mean(vals<=.05)),
                'unique_rounded_3dp':int(len(np.unique(np.round(vals,3))))}
            candidates=[]
            for other in G+E+O:
                if other==rule:continue
                x=f[other].to_numpy(dtype=float)
                if np.std(x)>1e-12 and np.std(vals)>1e-12:
                    cor=np.corrcoef(vals,x)[0,1]
                    if np.isfinite(cor):candidates.append((abs(float(cor)),float(cor),other))
            if candidates:
                v=max(candidates)
                rv.update(max_abs_pearson=v[0],most_redundant_feature=v[2],signed_pearson=v[1])
            else:rv.update(max_abs_pearson=np.nan,most_redundant_feature='',signed_pearson=np.nan)
            rows.append(rv)
        corr=f[O].corr(method='spearman')
        for i,aa in enumerate(O):
            for bb in O[i+1:]:
                pairs.append({'policy':policy,'dataset':ds,'window':w,'rule_a':aa,'rule_b':bb,
                    'spearman':float(corr.loc[aa,bb]) if np.isfinite(corr.loc[aa,bb]) else np.nan})
    save_csv(out,'ontology/rule_coverage_redundancy.csv',rows)
    save_csv(out,'ontology/rule_pairwise_correlation.csv',pairs)
    return rows

def verify_rdf(base,out,feat):
    """Independent rdflib parser checks rule confidence/provenance against source features."""
    ref.emit_ontology_rdf(out,feat,samples_per_dataset=14)
    tfile=out/'ontology_provenance_sample.ttl'
    g=rdflib.Graph();g.parse(tfile,format='turtle')
    EX=Namespace('https://example.org/rumordetection/')
    PROV=Namespace('http://www.w3.org/ns/prov#')
    catalog={v[0]:k for k,v in ref.RULE_DESC.items()}
    ix=feat[feat.window=='full'].set_index(['dataset','id'])
    checks=[];passed=0;failures=0
    for act in g.subjects(RDF.type,EX.RuleActivation):
        rr=list(g.objects(act,EX.rule));ss=list(g.objects(act,EX.confidence));origin=list(g.objects(act,PROV.wasDerivedFrom))
        ok=len(rr)==len(ss)==len(origin)==1 and g.value(origin[0],RDF.type)==EX.PropagationObservation
        if ok:
            rule_name=str(rr[0]).rsplit('/',1)[-1];rid=catalog.get(rule_name)
            refname=str(origin[0]).rsplit('thread_',1)[-1]
            ds,sourceid=refname.split('_',1)
            try:
                expected=float(ix.loc[(ds,sourceid),rid]); observed=float(ss[0])
                ok=bool(0<=observed<=1 and abs(expected-observed)<2e-5)
            except Exception:ok=False
        passed+=int(ok);failures+=int(not ok)
        checks.append({'activation':str(act),'confidence':float(ss[0]) if ss else np.nan,'provenance_valid':bool(ok)})
    save_csv(out,'ontology/rdf_validation_per_activation.csv',checks)
    summary={'rdf_triples':len(g),'activation_count':passed+failures,'validated':passed,'invalid':failures,
             'all_checks_passed':failures==0 and (passed+failures)>0,
             'scope':'sample of materialized numeric rule scores; not logical OWL entailment'}
    (out/'ontology/rdf_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if not summary['all_checks_passed']:raise AssertionError('RDF provenance audit FAILED: '+str(summary))
    return summary

def explain_full(D_model,xte,train,hold,pC,pD,thr,out,direction,policy):
    """Frozen model-local rule interventions: median substitution, NOT causal effect / proof."""
    if policy!='original':return []
    start=xte.shape[1]-len(O)
    med=np.median(train[O].to_numpy(dtype=float),axis=0)
    rows=[]
    for j,name in enumerate(O):
        x=xte.copy();x[:,start+j]=med[j]
        p0=D_model.predict_proba(x)[:,1]
        for i,record in enumerate(hold.itertuples(index=False)):
            rows.append({'direction':direction,'policy':policy,'id':str(record.id),
               'label':int(record.label),'rule':name,'observed_rule_score':float(getattr(record,name)),
               'prob_C':float(pC[i]),'prob_D':float(pD[i]),'prob_D_with_rule_median':float(p0[i]),
               'prob_intervention_delta':float(pD[i]-p0[i]),
               'prediction_C':int(pC[i]>=thr),'prediction_D':int(pD[i]>=thr),
               'annotation':'post-hoc model sensitivity; neither causal effect nor OWL entailment'})
    save_csv(out,f'explanations/local_rule_interventions_{direction}.csv',rows)
    # Top 20 changed-probability examples, record all materialized evidence scores.
    ranking=np.argsort(-np.abs(pD-pC))[:20]
    selected=[]
    for i in ranking:
        r=hold.iloc[i]
        selected.append({'direction':direction,'policy':policy,'id':str(r.id),
         'label':int(r.label),'threshold':float(thr),'prob_C':float(pC[i]),'prob_D':float(pD[i]),
         'delta_D_minus_C':float(pD[i]-pC[i]),'prediction_changed':bool((pD[i]>=thr)!=(pC[i]>=thr)),
         'rule_scores_json':json.dumps({k:round(float(r[k]),6) for k in O},sort_keys=True)})
    save_csv(out,f'explanations/top20_probability_changes_{direction}.csv',selected)
    return rows


def emit_prediction_provenance(out):
    """RDF explanation records for top model-change cases with linked rule-score evidence.
    Deliberately contains neither assertions of rumor truth nor OWL entailment.
    """
    from rdflib.namespace import PROV
    ex=Namespace('https://example.org/rumordetection/')
    gr=rdflib.Graph();gr.bind('ex',ex);gr.bind('prov',PROV)
    ncase=0;nscore=0
    for fp in sorted((out/'explanations').glob('top20_probability_changes_*.csv')):
        d=pd.read_csv(fp,dtype={'id':str})
        for j,row in enumerate(d.itertuples(index=False)):
            ds=row.direction.split('_to_')[1]
            thread=ex['thread_'+ds+'_'+str(row.id)]
            pred=ex['prediction_'+ds+'_'+str(row.id)]
            gr.add((thread,RDF.type,ex.PropagationObservation))
            gr.add((pred,RDF.type,ex.ModelPrediction))
            gr.add((pred,PROV.wasDerivedFrom,thread))
            gr.add((pred,PROV.wasGeneratedBy,ex.FrozenCatBoostOntology))
            gr.add((pred,ex.probabilityOntology,Literal(float(row.prob_D),datatype=XSD.double)))
            gr.add((pred,ex.probabilityRawEvidence,Literal(float(row.prob_C),datatype=XSD.double)))
            gr.add((pred,ex.threshold,Literal(float(row.threshold),datatype=XSD.double)))
            for rule,score in json.loads(row.rule_scores_json).items():
                b=rdflib.BNode()
                gr.add((pred,ex.hasEvidence,b));gr.add((b,RDF.type,ex.RuleScore))
                gr.add((b,ex.rule,ex[rule]));gr.add((b,ex.score,Literal(float(score),datatype=XSD.double)))
                gr.add((b,PROV.wasDerivedFrom,thread));nscore+=1
            ncase+=1
    dest=out/'ontology/prediction_evidence_provenance.ttl'
    gr.serialize(destination=str(dest),format='turtle')
    parsed=rdflib.Graph();parsed.parse(dest,format='turtle')
    valid=0
    for pred in parsed.subjects(RDF.type,ex.ModelPrediction):
        src=list(parsed.objects(pred,PROV.wasDerivedFrom))
        es=list(parsed.objects(pred,ex.hasEvidence))
        if len(src)==1 and len(es)==len(O) and all(
            len(list(parsed.objects(b,PROV.wasDerivedFrom)))==1 and
            0<=float(next(parsed.objects(b,ex.score)))<=1 for b in es):valid+=1
    summary={'n_prediction_explanations':ncase,'n_rule_score_evidence':nscore,
             'validated_predictions':valid,'total_rdf_triples':len(parsed),
             'valid':bool(valid==ncase),'note':'No full-window explanation requested' if ncase==0 else 'All model evidence entries validated'}
    (out/'ontology/prediction_provenance_validation.json').write_text(json.dumps(summary,indent=2))
    if not summary['valid']:raise AssertionError('Invalid prediction provenance '+str(summary))
    return summary

def holm_adjust(pvalues):
    # familywise conservative across multiple policy/window/direction comparisons
    a=np.array(pvalues,dtype=float);ids=np.argsort(a);res=np.empty_like(a);m=len(a);maxv=0
    for j,i in enumerate(ids):
        maxv=max(maxv,min(1.,a[i]*(m-j)));res[i]=maxv
    return res.tolist()

def eval_main(base,frame,features_by_policy,qa,blocks,out,windows,seed=42,placebos_full=49,placebos_early=9,bootstrap=1000):
    metrics=[];paired=[];placebos=[];predictions=[];started=time.time()
    ng=qa.set_index(['dataset','id'])
    for w in windows:
      for origin,target in DIRECTIONS:
        direction=f'{origin}_to_{target}';thr=FROZEN_C_THRESHOLDS[(direction,w)]
        for policy in POLICIES:
            F=features_by_policy[policy]
            if policy=='exclude_negative_trees':
                neg={(str(z.dataset),str(z.id)) for z in qa.itertuples(index=False) if int(z.issue_negative_delay)>0}
            else: neg=set()
            train=frame[(frame.dataset==origin)&(~frame.apply(lambda x:(x.dataset,x.id) in neg,axis=1))].copy()
            test=frame[(frame.dataset==target)&(~frame.id.isin(blocks[target]))&(~frame.apply(lambda x:(x.dataset,x.id) in neg,axis=1))].copy()
            train=train.merge(F[F.window==w],on=['dataset','id'],validate='one_to_one')
            test=test.merge(F[F.window==w],on=['dataset','id'],validate='one_to_one')
            train=train.reset_index(drop=True);test=test.reset_index(drop=True)
            if train.label.nunique()<2 or test.label.nunique()<2:raise RuntimeError('One-class split: '+direction+' '+policy)
            xt,xv,_=ref.build_text_encoder(train.source_text,test.source_text)
            pC=fit_model(train,test,xt,xv,G+E,seed=seed)
            pD,mdl,xtr,xte=fit_model(train,test,xt,xv,G+E+O,seed=seed,return_matrix=True)
            metC=ref.base_metric(test.label,pC,thr);metD=ref.base_metric(test.label,pD,thr)
            for k,met in [('C_RawEvidence',metC),('D_RealOntology',metD)]:
                metrics.append({'policy':policy,'window':w,'direction':direction,'config':k,
                  'n_train':len(train),'n_test':len(test),'threshold_frozen_C_source_oof':thr,**met})
            ci=bootstrap_diff(test.label,pD,pC,thr,bootstrap,seed+len(paired))
            paired.append({'policy':policy,'window':w,'direction':direction,'n_test':len(test),
                'macro_f1_C':metC['macro_f1'],'macro_f1_D':metD['macro_f1'],
                'delta_D_minus_C':metD['macro_f1']-metC['macro_f1'],
                'delta_rumor_recall':metD['rumor_recall']-metC['rumor_recall'],
                'delta_FP':metD['fp']-metC['fp'],'delta_FN':metD['fn']-metC['fn'],
                'bootstrap_ci_025':ci[0],'bootstrap_ci_975':ci[1],
                'ci_method':'paired percentile bootstrap on target threads, exploratory'})
            for i,r in enumerate(test.itertuples(index=False)):
                predictions.append({'policy':policy,'window':w,'direction':direction,'dataset':target,'id':str(r.id),
                    'label':int(r.label),'p_C':float(pC[i]),'p_D':float(pD[i]),'threshold':thr,
                    'pred_C':int(pC[i]>=thr),'pred_D':int(pD[i]>=thr)})
            if w=='full':
                explain_full(mdl,xte,train,test,pC,pD,thr,out,direction,policy)
            # Placebo is the primary original policy only; never use target labels for matching.
            if policy=='original':
                repeats=placebos_full if w=='full' else placebos_early
                for rep in range(repeats):
                    pp=fit_model(train,test,xt,xv,G+E+O,seed=seed,placebo_seed=seed+100001+rep)
                    fm=ref.base_metric(test.label,pp,thr)['macro_f1']
                    placebos.append({'window':w,'direction':direction,'policy':policy,'repeat':rep,
                        'placebo_macro_f1':fm,'D_macro_f1':metD['macro_f1'],
                        'C_macro_f1':metC['macro_f1'],'threshold':thr})
            save_csv(out,'metrics_by_policy.csv',metrics)
            save_csv(out,'paired_C_D_bootstrap.csv',paired)
            save_csv(out,'placebo_retrained.csv',placebos)
            save_csv(out,'predictions_by_thread.csv',predictions)
            print('DONE',policy,direction,w,'n=',len(test),'Δ(D-C)=',round(metD['macro_f1']-metC['macro_f1'],4),
                  'elapsed_min=',round((time.time()-started)/60,1),flush=True)
    pp=pd.DataFrame(placebos);tests=[]
    for (w,d),f in pp.groupby(['window','direction']):
        obs=float(f.D_macro_f1.iloc[0]); vals=f.placebo_macro_f1.to_numpy(dtype=float)
        tests.append({'window':w,'direction':d,'repeats':len(vals),'real_D_macro_f1':obs,
            'placebo_mean':float(vals.mean()),'placebo_std':float(vals.std(ddof=1)) if len(vals)>1 else 0,
            'montecarlo_one_sided_p':float((1+np.sum(vals>=obs))/(len(vals)+1)),
            'interpretation':'conditional permutation null, exploratory; 9-repeat early windows low p resolution'})
    tests=pd.DataFrame(tests)
    tests['holm_adjusted_p_over_eight_tests']=holm_adjust(tests.montecarlo_one_sided_p.to_numpy())
    tests.to_csv(out/'placebo_statistics.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics_by_policy.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_C_D_bootstrap.csv',index=False)
    return metrics,paired,tests

def explain_audit(out,rule_rows):
    ed=[]
    for f in out.glob('explanations/local_rule_interventions_*.csv'):
        d=pd.read_csv(f)
        for rule,g in d.groupby('rule'):
            delta=g.prob_intervention_delta.to_numpy(dtype=float)
            ed.append({'direction':g.direction.iloc[0],'rule':rule,'n':len(g),
             'mean_abs_probability_delta':float(np.mean(np.abs(delta))),
             'fraction_changes_ge_001':float(np.mean(np.abs(delta)>=.01)),
             'mean_signed_probability_delta':float(np.mean(delta))})
    save_csv(out,'explanations/rule_sensitivity_summary.csv',ed)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='/kaggle/working/Twitter1516_Sensitivity_Ontology_Audit_Results')
    ap.add_argument('--workdir',default='/kaggle/working/twitter1516_full_raw')
    ap.add_argument('--input-root',default='/kaggle/input')
    ap.add_argument('--base-dir',default='')
    ap.add_argument('--reference-results-zip',default='')
    ap.add_argument('--no-download',action='store_true')
    ap.add_argument('--test-fixture',action='store_true',help='Only for technical tests; never used for research reporting')
    ap.add_argument('--windows',nargs='+',default=WINDOWS)
    ap.add_argument('--placebos-full',type=int,default=49)
    ap.add_argument('--placebos-early',type=int,default=9)
    ap.add_argument('--bootstrap',type=int,default=1000)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for sub in ['audit','ontology','explanations']: (out/sub).mkdir(exist_ok=True)
    if args.test_fixture:ref.EXPECTED={'twitter15':72,'twitter16':72}
    base=Path(args.base_dir) if args.base_dir else ref.prepare_data(Path(args.workdir),Path(args.input_root),not args.no_download)
    frame,inventory=ref.load_datasets(base,strict=True)
    inventory.to_csv(out/'audit/inventory.csv',index=False)
    summary,blocks=ref.duplicate_audit(frame,out/'audit')
    features,issues=materialize_features(frame,base,out,list(dict.fromkeys(args.windows+['full'])))
    rule_audit=build_rule_audit(features,out)
    rdf=verify_rdf(base,out,features['original'])
    fcfg={'protocol':'Frozen Twitter15/16 Sensitivity + Structural Fuzzy Ontology Audit','version':'2026-11-05',
        'seed':SEED,'policies':POLICIES,'windows':args.windows,
        'reference_thresholds_C_source_only_oof':{f'{x[0]}::{x[1]}':y for x,y in FROZEN_C_THRESHOLDS.items()},
        'original_repo_git_sha':ref.ORIGINAL_SHA,'text_mirror_git_sha':ref.SOURCE_SHA,
        'metric':'binary rumor detection; true/false/unverified vs non-rumor',
        'placebos_full':args.placebos_full,'placebos_early':args.placebos_early,
        'bootstrap':args.bootstrap,'reference_test_data_seen_during_model_development':True,
        'true_external_validation_completed':False,
        'ontology_rule_type':'fixed label-blind structural fuzzy scores; NOT formal OWL-DL entailment',
        'model_architecture':'CatBoost frozen params from previous pipeline',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'previous_pipeline_sha256':hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
        'test_fixture':args.test_fixture}
    (out/'FROZEN_PROTOCOL.json').write_text(json.dumps(fcfg,ensure_ascii=False,indent=2),encoding='utf-8')
    met,paired,tests=eval_main(base,frame,features,issues,blocks,out,args.windows,seed=SEED,
      placebos_full=args.placebos_full,placebos_early=args.placebos_early,bootstrap=args.bootstrap)
    explain_audit(out,rule_audit)
    prediction_rdf_summary=emit_prediction_provenance(out)
    (out/'THIRD_DATASET_STATUS.txt').write_text(
      'NOT PERFORMED. Twitter15 and Twitter16 have both been used during iterative development. '
      'This notebook is a sensitivity and evidence audit; it is not independent third-dataset validation. '
      'No claims of untouched independent confirmation should be made.\n',encoding='utf-8')
    if args.reference_results_zip:
        try:
            import io
            with zipfile.ZipFile(args.reference_results_zip) as z:
                orig=pd.read_csv(io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith('/predictions.csv')))),dtype={'id':str})
                our=pd.read_csv(out/'predictions_by_thread.csv',dtype={'id':str})
                bl=our[our.policy=='original']
                ids=['direction','window','id']
                original_C=orig[orig.config=='C'][ids+['probability']].rename(columns={'probability':'previous_p_C'})
                original_D=orig[orig.config=='D'][ids+['probability']].rename(columns={'probability':'previous_p_D'})
                m=bl.merge(original_C,on=ids,how='left').merge(original_D,on=ids,how='left')
                m['absdiff_C']=(m.p_C-m.previous_p_C).abs();m['absdiff_D']=(m.p_D-m.previous_p_D).abs()
                m[ids+['absdiff_C','absdiff_D']].to_csv(out/'audit/previous_run_prediction_reproduction.csv',index=False)
                s={'n_matched':int(m.previous_p_C.notna().sum()),'n_all':len(m),
                   'max_absdiff_C':float(m.absdiff_C.max()),'max_absdiff_D':float(m.absdiff_D.max())}
                (out/'audit/previous_run_reproduction.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
                print('Previous run check:',s,flush=True)
        except Exception as exc:
            (out/'audit/previous_run_reproduction_error.txt').write_text(str(exc))
    report=[]
    report.append('# Twitter15/Twitter16 — Sensitivity + Ontology Evidence Audit')
    report.append('Run completed with frozen C source-OOF thresholds. These datasets were used during method development; NOT untouched independent validation.')
    report.append('## Ontology scope\nLabel-blind, structural fuzzy evidence rules; not truth verification, not reply-stance inference and not OWL-DL entailment.')
    report.append('## Data and tree issues\n'+(out/'audit/tree_policy_summary.csv').read_text()[:3000])
    ppd=pd.DataFrame(paired)
    report.append('## D-C paired comparisons (same threshold)\n'+ppd[['policy','window','direction','n_test','delta_D_minus_C','bootstrap_ci_025','bootstrap_ci_975']].to_string(index=False))
    report.append('## Placebo p and multiplicity\n'+tests.to_string(index=False))
    report.append('## RDF sample provenance validation\n'+json.dumps(rdf,indent=2))
    report.append('## RDF model prediction provenance validation\n'+json.dumps(prediction_rdf_summary,indent=2))
    report.append('## Interpretability\nPost-hoc rule-median substitution measures model sensitivity only, not a causal effect. '
                  'Use rule_coverage_redundancy.csv and the local intervention files; check stability under every tree policy.')
    report.append('## Research decision\nPredeclare original policy as primary. Strict and negative-tree exclusion are sensitivity only; do not select best test score. '
       'Early-window placebo has lower Monte-Carlo p-value resolution. Interpret all tests as exploratory due to repeated benchmark use.')
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    zpath=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in out.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(out.parent))
    print('FINAL ZIP:',zpath,'size:',zpath.stat().st_size,'bytes',flush=True)

if __name__=='__main__':main()
