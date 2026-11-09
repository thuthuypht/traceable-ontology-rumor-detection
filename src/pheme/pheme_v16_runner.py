"""PHEME v16: strong v14 ontology rescue + independent CatBoost, reliability-gated fusion.

This is a new DEVELOPMENT experiment; PHEME has been repeatedly used. No test
labels may influence training, thresholding, feature selection, or model choice.
All evaluations are retrospective full-thread classification.
"""
from __future__ import annotations
import argparse, os, json, hashlib, zipfile, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import pheme_v14_pipeline as base
import pheme_v11_features as old

SHAPE=['t_source_length','t_source_wordcount','t_source_qmarks',
       't_source_hashtagcount','t_source_links','t_source_exclamation']
FEATURES={
 'B_Graph':SHAPE+old.GRAPH,
 'C_Graph_RawEvidence':SHAPE+old.GRAPH+old.RAW,
 'D_OntologyV11_LGBM':SHAPE+old.GRAPH+old.RAW+old.RULES,
 'E_OntologyV11_CatBoost':SHAPE+old.GRAPH+old.RAW+old.RULES,
}
FAMILIES=['A_TextFusion',*FEATURES,'F_Ontology_Ensemble','H_Ontology_ReliabilityGate']
SCORES=['MacroF1','Recall','Precision','FN','FP','AUPRC']


def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def nonleaky_reliability(df):
 # Deterministic, label-blind and event-blind confidence proxy. Not a factual truth score.
 cv=np.clip(pd.to_numeric(df['e_reply_coverage'],errors='coerce').fillna(0).to_numpy(float),0,1)
 diversity=np.clip(pd.to_numeric(df['e_dispute_diversity'],errors='coerce').fillna(0).to_numpy(float),0,1)
 n=np.maximum(0,pd.to_numeric(df['source_reply_count'],errors='coerce').fillna(0).to_numpy(float))
 magnitude=1-np.exp(-n/5)
 return np.clip(.20+.80*(.45*cv+.30*diversity+.25*magnitude),.2,1.).astype('float32')


def base_choices(y,ev,rows,raw):
 opts=[(1.,0.,0.),(.7,.3,0.),(.4,.6,0.),(.7,0.,.3),(.4,.3,.3),(.2,.4,.4),(0.,1.,0.),(0.,0.,1.)]
 picks=[]
 for ws,wc,wm in opts:
  p=ws*raw['source'][rows]+wc*raw['context'][rows]+wm*raw['minilm'][rows]
  threshold,score=base.tune(y[rows],p,ev[rows])
  picks.append((score,-abs(threshold-.5),ws,(ws,wc,wm),threshold,p))
 _,_,_,weights,t,pred=max(picks,key=lambda a:a[:3])
 return {'weights':weights,'threshold':t,'score':float(base.tune(y[rows],pred,ev[rows])[1]),'prob':pred}


def choose_alpha(y,ev,idx,base_p,branch_p,min_gain):
 idx=np.asarray(idx);best_t,base_s=base.tune(y[idx],base_p,ev[idx]);
 per={e:base.metrics(y[idx][ev[idx]==e],base_p[ev[idx]==e],best_t)['MacroF1']
      for e in np.unique(ev[idx])}
 choice=dict(alpha=0.,threshold=best_t,score=base_s,mean_gain=0.,event_consistency=0)
 for alpha in (.15,.30,.50,.70,1.):
  p=(1-alpha)*base_p+alpha*branch_p
  t,s=base.tune(y[idx],p,ev[idx]); gains=[]
  for e in np.unique(ev[idx]):
   ix=ev[idx]==e
   gains.append(base.metrics(y[idx][ix],p[ix],t)['MacroF1']-per[e])
  # Check 3 of 4 held-out inner events are not substantially harmed.
  agrees=int(sum(g>=-.003 for g in gains));min_agree=max(1,int(np.ceil(.60*len(gains))))
  if s>=base_s+min_gain and agrees>=min_agree and s>choice['score']+1e-9:
   choice=dict(alpha=float(alpha),threshold=float(t),score=float(s),
               mean_gain=float(np.mean(gains)),event_consistency=agrees)
 return choice


def fit_fold(df,sem,txt_s,txt_c,char_s,char_c,y,ev,tr,te,seed,fast,permute_test=False):
 out={}
 out['source']=base.clf_text(tr,te,txt_s,y,ev,char_s,seed)
 out['context']=base.clf_text(tr,te,txt_c,y,ev,char_c,seed)
 out['minilm']=base.clf_emb(tr,te,sem['src'],y,ev,seed) if sem['src'] is not None else out['source']
 proj=base.train_only_projection(tr,te,sem['embedding'],txt_s,txt_c,fast=fast)
 for name,cols in FEATURES.items():
  variant='catboost' if 'CatBoost' in name else 'v12'
  pp,m=base.clf_dense(tr,te,df,cols,sem['embedding'],y,ev,seed,low_memory=fast,
                      projection=proj,variant=variant)
  out[name]=pp
  if permute_test and name=='D_OntologyV11_LGBM':
   # Post-hoc test sensitivity ONLY: hold semantic, graph, and raw evidence fixed.
   # Shuffle ONLY ontology rule columns. Never used for training/model selection.
   selected=list(m._v13_selected)
   xt=df.iloc[te][selected].to_numpy(dtype=np.float32).copy()
   rng=np.random.default_rng(1234)
   ix=rng.permutation(len(te))
   for rule in old.RULES:
    if rule in selected:
     c=selected.index(rule);xt[:,c]=xt[ix,c]
   xx=np.column_stack([proj[1],xt])
   out['__permuted_ontology_rules']=m.predict_proba(pd.DataFrame(
       xx,columns=[f'f_{j}' for j in range(xx.shape[1])]))[:,1].astype('float32')
 out['F_Ontology_Ensemble']=.5*(out['D_OntologyV11_LGBM']+out['E_OntologyV11_CatBoost'])
 rel=nonleaky_reliability(df)[te]
 out['H_Ontology_ReliabilityGate']=(out['C_Graph_RawEvidence']+rel*(out['D_OntologyV11_LGBM']-out['C_Graph_RawEvidence']))
 return out


def run(args):
 np.random.seed(args.seed);random.seed(args.seed)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 data,emb,src,reps=base.prepare_all(args)
 y=data.label.to_numpy(int);ev=data.event_id.astype(str).to_numpy()
 assert y.min()==0 and y.max()==1 and len(np.unique(ev))==5
 # Label-free audit of exact normalized source-text collisions across events.
 text_normalized=(data.source_text.fillna('').astype(str).str.lower()
                   .str.replace(r'\s+',' ',regex=True).str.strip())
 event_counts=pd.DataFrame({'text_norm':text_normalized,'event':ev}).groupby('text_norm').event.nunique()
 collisions=set(event_counts[event_counts>1].index)-{''}
 audit=pd.DataFrame([{'source_text_normalized':t,'n_distinct_events':int(event_counts[t])}
                     for t in sorted(collisions)])
 audit.to_csv(out/'cross_event_source_text_overlap_audit.csv',index=False)
 print('Exact source-text types appearing in multiple events:',len(collisions),flush=True)
 assert data.thread_id.is_unique and not data[FEATURES['D_OntologyV11_LGBM']].isna().any().any()
 r1=nonleaky_reliability(data)
 flipped=data.copy();flipped['label']=1-flipped.label;flipped['event_id']='FAKE'
 np.testing.assert_array_equal(r1,nonleaky_reliability(flipped))
 assert all(not any(x in col.lower() for x in ('label','rumor_ratio','user_prior','event_id'))
            for cols in FEATURES.values() for col in cols)
 source_txt,context_txt=base.make_texts(data,reps)
 charvec=HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**16,
                           alternate_sign=False,norm='l2',dtype=np.float32)
 char_s=charvec.transform(source_txt);char_c=charvec.transform(context_txt)
 sem={'src':src,'embedding':emb}
 all_events=sorted(np.unique(ev).tolist())
 selected=all_events if args.events=='all' else [x.strip() for x in args.events.split(',')]
 if any(e not in all_events for e in selected):raise ValueError(f'Unknown events: {selected}')
 fingerprint={
  'protocol':'v16_frozen_v11_rule_ensemble_1','seed':args.seed,'fast':bool(args.fast),
  'min_gain':args.min_gain,'max_replies':args.max_replies,'no_minilm':bool(args.no_minilm),
  'data_sha256':sha256(args.pheme),'graph_sha256':sha256(args.graph),
  'code_sha256':{f:sha256(Path(__file__).with_name(f)) for f in
      ['pheme_v16_runner.py','pheme_v11_features.py','pheme_v14_rules.py','pheme_v14_pipeline.py']}}
 signature=out/'experiment_signature.json'
 if any(out.glob('fold_*.json')):
  if not signature.exists() or json.loads(signature.read_text())!=fingerprint:
   raise ValueError('Checkpoint fingerprint mismatch: choose a new OUTPUT_PREFIX')
 signature.write_text(json.dumps(fingerprint,indent=2))
 for event in selected:
  ck=out/f'fold_{event}.json'
  if ck.exists():print('Resuming completed event',event,flush=True);continue
  te=np.where(ev==event)[0];tr=np.where(ev!=event)[0]
  print('\nOUTER HELD-OUT EVENT:',event,'train',len(tr),'test',len(te),flush=True)
  oof={k:np.full(len(data),np.nan,dtype='float32') for k in ['source','context','minilm',*FAMILIES[1:]]}
  for j,inner in enumerate(sorted(np.unique(ev[tr]))):
   itr=tr[ev[tr]!=inner];iva=tr[ev[tr]==inner]
   print('  inner held out',inner,len(itr),len(iva),flush=True)
   fold=fit_fold(data,sem,source_txt,context_txt,char_s,char_c,y,ev,itr,iva,args.seed+j,args.fast)
   for k,v in fold.items():oof[k][iva]=v
  assert all(np.isfinite(oof[k][tr]).all() for k in oof)
  bc=base_choices(y,ev,tr,oof)
  text_oof=bc['prob'];decisions={'A_TextFusion':{'alpha':0.,'threshold':bc['threshold'],'score':bc['score']}}
  for name in FAMILIES[1:]:decisions[name]=choose_alpha(y,ev,tr,text_oof,oof[name][tr],args.min_gain)
  penalty={'A_TextFusion':0.,'B_Graph':.0002,'C_Graph_RawEvidence':.0005,
           'D_OntologyV11_LGBM':.0008,'E_OntologyV11_CatBoost':.001,
           'F_Ontology_Ensemble':.0012,'H_Ontology_ReliabilityGate':.0013}
  selected_family=max(FAMILIES,key=lambda n:(decisions[n]['score']-penalty[n],-penalty[n]))
  full=fit_fold(data,sem,source_txt,context_txt,char_s,char_c,y,ev,tr,te,args.seed,args.fast,permute_test=True)
  ws,wc,wm=bc['weights']; text_test=ws*full['source']+wc*full['context']+wm*full['minilm']
  rows=[];pred=[]
  for name in FAMILIES:
   sel=decisions[name]
   p=text_test if name=='A_TextFusion' else (1-sel['alpha'])*text_test+sel['alpha']*full[name]
   metric=base.metrics(y[te],p,sel['threshold'])
   rows.append({'event':event,'Config':name,'alpha':sel['alpha'],
                'threshold':sel['threshold'],'inner_score':sel['score'],**metric})
   pred.extend([{'event':event,'Config':name,'thread_id':str(data.iloc[idx].thread_id),
                 'y_true':int(y[idx]),'p_rumor':float(pp),'threshold':float(sel['threshold'])}
                 for idx,pp in zip(te,p)])
  result=[r for r in rows if r['Config']==selected_family][0].copy()
  result['Config']='G_EventCV_Selected';result['chosen_family']=selected_family
  rows.append(result)
  pred.extend([dict(x,Config='G_EventCV_Selected') for x in pred if x['Config']==selected_family])
  # Diagnostics: fixed alpha with threshold selected on inner events only, NOT on test.
  diagnosis=[]
  for name in ['C_Graph_RawEvidence','D_OntologyV11_LGBM','E_OntologyV11_CatBoost',
               'F_Ontology_Ensemble','H_Ontology_ReliabilityGate']:
   for w in (.5,1.):
    pin=(1-w)*text_oof+w*oof[name][tr]; t,_=base.tune(y[tr],pin,ev[tr]);
    p=(1-w)*text_test+w*full[name]
    diagnosis.append({'event':event,'Config':name,'forced_alpha':w,
       'threshold_train_only':t,**base.metrics(y[te],p,t)})
  # Predeclared post-hoc rule-permutation sensitivity. It NEVER selects a model.
  onto=decisions['D_OntologyV11_LGBM']
  pp_real=(1-onto['alpha'])*text_test+onto['alpha']*full['D_OntologyV11_LGBM']
  pp_permuted=(1-onto['alpha'])*text_test+onto['alpha']*full['__permuted_ontology_rules']
  neg={'event':event,'OntologyV11_MacroF1_real':base.metrics(y[te],pp_real,onto['threshold'])['MacroF1'],
       'OntologyV11_MacroF1_permuted_rules':base.metrics(y[te],pp_permuted,onto['threshold'])['MacroF1'],
       'alpha':onto['alpha'], 'mean_absolute_probability_change':float(np.mean(np.abs(pp_real-pp_permuted))),
       'note':'Test-set sensitivity diagnostic ONLY; frozen model, shuffled rule columns, NOT a fully retrained placebo'}
  pack=dict(event=event,metrics=rows,predictions=pred,selection={'event':event,
       'selected_family':selected_family,'base_weights':list(bc['weights']),
       'base_threshold':bc['threshold'],'choices':decisions},forced=diagnosis,negative_control=neg)
  temp=ck.with_suffix('.tmp');temp.write_text(json.dumps(pack));os.replace(temp,ck)
  print(' event selected:',selected_family,'macro-F1',round(result['MacroF1'],4),flush=True)
 # Rebuild outputs only from committed per-event checkpoints. Never double count.
 foldpacks=[json.loads((out/f'fold_{e}.json').read_text()) for e in selected if (out/f'fold_{e}.json').exists()]
 m=pd.DataFrame([r for f in foldpacks for r in f['metrics']])
 p=pd.DataFrame([r for f in foldpacks for r in f['predictions']])
 assert not m.duplicated(['event','Config']).any()
 assert not p.duplicated(['event','Config','thread_id']).any()
 m.to_csv(out/'metrics_per_event.csv',index=False)
 p.to_csv(out/'test_predictions_by_thread.csv',index=False)
 pd.DataFrame([f['selection'] for f in foldpacks]).to_json(out/'inner_selection_audit.json',orient='records',indent=2)
 pd.DataFrame([r for f in foldpacks for r in f['forced']]).to_csv(out/'forced_branch_diagnostics.csv',index=False)
 pd.DataFrame([f['negative_control'] for f in foldpacks]).to_csv(out/'ontology_sensitivity_diagnostics.csv',index=False)
 summaries=m.groupby('Config').agg(MacroF1_mean=('MacroF1','mean'),MacroF1_std=('MacroF1','std'),
    RumorRecall_mean=('Recall','mean'),AUPRC_mean=('AUPRC','mean'),FN_total=('FN','sum'),
    FP_total=('FP','sum'),n_events=('event','nunique')).reset_index()
 summaries.to_csv(out/'summary_by_config.csv',index=False)
 (out/'README_RESULTS.md').write_text('PHEME v16: frozen v11 ontology, LGBM and CatBoost with reliability-gated fusion.\n'
    'All model selection and thresholds use inner held-out training events ONLY.\n'
    'Avoid test-driven selection; new external dataset needed for publication.\n'
    'A Text, B Graph, C Raw Evidence, D/E Ontology LGBM/CatBoost, F 50/50 ontology ensemble, H Reliability-Gated, G inner-selected.\n'
    'Retrospective thread detection; heuristic rule provenance is NOT verified truth or OWL-DL entailment.\n')
 print(summaries.to_string(index=False),flush=True)
 zp=out.with_suffix('.zip')
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
  for item in out.glob('*'):
   if item.is_file() and item.suffix!='.tmp':z.write(item,arcname=item.name)
 print('RESULT ZIP:',zp,flush=True)
 return zp


def main():
 ap=argparse.ArgumentParser()
 for arg in ('pheme','graph','output','cache'):ap.add_argument('--'+arg,required=True)
 ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--max-replies',type=int,default=8)
 ap.add_argument('--batch-size',type=int,default=64)
 ap.add_argument('--events',default='all')
 ap.add_argument('--fast',action='store_true')
 ap.add_argument('--no-minilm',action='store_true')
 ap.add_argument('--min-gain',type=float,default=.002)
 ap.add_argument('--resume',action='store_true')
 args=ap.parse_args()
 if args.max_replies!=8:raise ValueError('For comparability v16 requires MAX_REPLIES=8')
 args.cache_dir=args.cache
 run(args)

if __name__=='__main__':main()
