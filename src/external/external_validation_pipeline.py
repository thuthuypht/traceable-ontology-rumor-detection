"""Twitter15+16 -> Twitter-COVID19 target-label-blind external test.
Primary: full-tree harmonized structural features; source-only grouped 3-fold CV;
matched CatBoost models; retrained conditional ontology placebos and bootstrap.
Target labels are never accessed in training, threshold selection or model selection.
Published data may have ambiguous delay units; no hour-based claims are made.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,re,sys,time,zipfile
from pathlib import Path
from collections import Counter,defaultdict,deque
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, average_precision_score, precision_score, recall_score,accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from frozen_twitter1516_pipeline import (prepare_data,load_datasets,parse_tree,groupids,norm_text,build_text_encoder,make_predictor,select_threshold)
from twitter_covid19_preflight import URL as COVID_URL, SHA as COVID_SHA, download_archive, archive_extract, sha256

SEED=42
COVID_EXPECTED=400
EXPECTED_ARCHIVE_SHA='51fb8799aa6819990d75667b12d459ba869387f3d8ca5ba156f678832c95333f'
G=['g_log_nodes','g_log_edges','g_root_degree','g_max_depth','g_mean_depth','g_leaf_ratio','g_breadth','g_depth2_ratio']
R=['e_delay_median_norm','e_delay_p90_norm','e_delay_iqr_norm','e_frac_first10pct','e_frac_first25pct','e_frac_first50pct','e_parent_lag_norm','e_root_branch_ratio','e_temporal_entropy','e_late_quartile','e_early_burst']
O=['o_deep_persistence','o_delayed_revival','o_temporal_concentration','o_branching_persistence']
MODELS={'A':[], 'B':G, 'C':G+R,'D':G+R+O}


def simple_file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def root_reachable_tree(nodes,edges,root):
 """Filter temporally backward, negative, self and disconnected nodes.
 Node keys are unique *node occurrence keys*, never assumed unique tweet IDs.
 """
 good={k:float(t) for k,t in nodes.items() if np.isfinite(t) and t>=0}
 if root not in good:return [],{}, {'unreachable_or_invalid':len(nodes),'valid_edges':0}
 adj=defaultdict(list);bad=Counter()
 for a,b in edges:
  if a==b:bad['self_edge']+=1;continue
  if a not in good or b not in good:bad['missing_or_negative_node']+=1;continue
  if good[b]<good[a]-1e-8:bad['backward_edge']+=1;continue
  adj[a].append(b)
 parent={};depth={root:0};q=deque([root])
 while q:
  u=q.popleft()
  for v in sorted(set(adj[u]),key=lambda x:(good[x],str(x))):
   if v==root or v in depth:continue
   parent[v]=u;depth[v]=depth[u]+1;q.append(v)
 bad['unreachable_or_invalid']=len(nodes)-len(depth)
 bad['valid_edges']=len(parent)
 return [(p,c) for c,p in parent.items()],{k:good[k] for k in depth},dict(bad)


def source_tree(path,source_id):
    """Use actual source-tweet node as root, without a synthetic extra node.
    This yields the same root-node semantics as COVID parent=None/index=1.
    """
    obj=parse_tree(path)
    real=[n for n in obj['nodes'] if n[1]!='ROOT']
    if not real:raise RuntimeError('Source tree has no real nodes '+str(path))
    roots=[b for a,b in obj['edges'] if a[1]=='ROOT' and b[1]!='ROOT']
    if not roots:raise RuntimeError('Source tree lacks ROOT edge '+str(path))
    roots.sort(key=lambda n:(n[2],n[0],n[1]))
    root=roots[0]
    node_time={n:float(n[2]) for n in real}
    edges=[(a,b) for a,b in obj['edges'] if a[1]!='ROOT' and b[1]!='ROOT']
    for other in roots[1:]:edges.append((root,other))
    return node_time,edges,root,obj['issues']


def parse_covid_files(root:Path):
 labels=list(root.rglob('Twitter_label_all.txt'));data=list(root.rglob('Twitter_data_all.txt'))
 if len(labels)!=1 or len(data)!=1:raise RuntimeError(f'Expected one COVID label/data file each, got {len(labels)}/{len(data)}')
 label_map={};bad=0
 for line in labels[0].read_text(encoding='utf-8-sig',errors='replace').splitlines():
  p=line.strip().split('\t')
  if len(p)!=2 or p[1].strip() not in {'0','1'}:bad+=1;continue
  key=p[0].strip()
  if key in label_map:raise RuntimeError('Duplicate COVID label ID '+key)
  label_map[key]=int(p[1].strip())
 if bad or len(label_map)!=COVID_EXPECTED:raise RuntimeError(f'Invalid COVID labels n={len(label_map)} bad={bad}')
 trees=defaultdict(dict); invalid=Counter();seen_source={}
 # Number of posts is large (~400k), streaming parse avoids retaining all reply text.
 with data[0].open(encoding='utf-8-sig',errors='replace') as f:
  for i,line in enumerate(f,1):
   p=line.rstrip('\r\n').split('\t',4)
   if len(p)!=5:invalid['bad_column_count']+=1;continue
   eid,par,nid,delay,text=[x.strip() for x in p]
   if eid not in label_map:invalid['event_id_not_in_labels']+=1;continue
   try:dt=float(delay)
   except ValueError:invalid['bad_delay']+=1;continue
   if not np.isfinite(dt):invalid['nonfinite_delay']+=1;continue
   if nid in trees[eid]:invalid['duplicate_node_index']+=1;continue
   parent=par if par.lower() not in {'none','null','root','-1'} else None
   trees[eid][nid]=(parent,dt)
   if parent is None:
    if eid in seen_source:invalid['multiple_root_rows']+=1
    else:seen_source[eid]=text
 rows=[];graph={}
 for eid,y in label_map.items():
  n=trees.get(eid,{})
  roots=[idx for idx,(parent,delay) in n.items() if parent is None]
  if len(roots)!=1:invalid['event_invalid_root_count']+=1;continue
  root=roots[0];node_t={idx:delay for idx,(par,delay) in n.items()}
  edges=[(parent,idx) for idx,(parent,delay) in n.items() if parent is not None]
  graph[eid]=(node_t,edges,root, {'raw_nodes':len(n),'raw_edges':len(edges)})
  tx=seen_source.get(eid,'').strip()
  if not tx:invalid['missing_root_text']+=1;continue
  rows.append({'dataset':'twitter_covid19','id':eid,'source_text':tx,'normalized':norm_text(tx),'label':y})
 df=pd.DataFrame(rows)
 if len(df)<int(.95*COVID_EXPECTED):raise RuntimeError(f'Too many unusable target trees: {len(df)} / {COVID_EXPECTED} ({dict(invalid)})')
 return df,graph,dict(invalid),data[0],labels[0]


def features_for_tree(node_t,edge_list,root,window='full'):
 # Time units cancel by using fractions of each tree's *observed duration*.
 if window!='full':raise ValueError('Main protocol uses full tree only; delay unit not verified across corpora')
 edges,valid,issues=root_reachable_tree(node_t,edge_list,root)
 tmax=max(valid.values(),default=0.)
 normtimes={k:((v-valid[root])/max(tmax-valid[root],1e-9)) for k,v in valid.items()}
 parent={v:u for u,v in edges};children=defaultdict(list)
 for u,v in edges:children[u].append(v)
 depth={root:0};q=deque([root])
 while q:
  u=q.popleft()
  for v in children[u]:depth[v]=depth[u]+1;q.append(v)
 replies=[k for k in valid if k!=root]; nr=len(replies);nn=len(valid)
 times=np.array([min(1.,max(0.,normtimes[k])) for k in replies],dtype=float)
 dvals=np.array([depth[k] for k in replies],dtype=float)
 outdeg=[len(children[k]) for k in replies]
 root_degree=len(children[root])
 bt=Counter(int(min(x,.999999)*5) for x in times)
 prob=np.array(list(bt.values()),float)/max(nr,1)
 ent=float(-(prob*np.log(prob)).sum()/np.log(5)) if len(prob)>1 else 0.
 lag=np.array([max(0.,normtimes[v]-normtimes[u]) for u,v in edges],dtype=float)
 med=float(np.median(times)) if nr else 0.
 q90=float(np.quantile(times,.90)) if nr else 0.
 q75=float(np.quantile(times,.75)) if nr else 0.
 q25=float(np.quantile(times,.25)) if nr else 0.
 e10=float(np.mean(times<=.10)) if nr else 0.
 e25=float(np.mean(times<=.25)) if nr else 0.
 e50=float(np.mean(times<=.50)) if nr else 0.
 late=float(np.mean(times>=.75)) if nr else 0.
 vals={
 'g_log_nodes':float(np.log1p(nn)), 'g_log_edges':float(np.log1p(len(edges))),
 'g_root_degree':float(np.log1p(root_degree)), 'g_max_depth':float(np.log1p(max(depth.values(),default=0))),
 'g_mean_depth':float(dvals.mean()) if nr else 0.,'g_leaf_ratio':float(np.mean(np.array(outdeg)==0)) if nr else 1.,
 'g_breadth':float(np.log1p(max(Counter(depth.values()).values(),default=0))),
 'g_depth2_ratio':float(np.mean(dvals>=2)) if nr else 0.,
 'e_delay_median_norm':med,'e_delay_p90_norm':q90,'e_delay_iqr_norm':q75-q25,
 'e_frac_first10pct':e10,'e_frac_first25pct':e25,'e_frac_first50pct':e50,
 'e_parent_lag_norm':float(np.median(lag)) if len(lag) else 0.,
 'e_root_branch_ratio':root_degree/max(nn,1),'e_temporal_entropy':ent,'e_late_quartile':late,
 'e_early_burst':float(e10*np.log1p(nr)),
 'o_deep_persistence':float(expit(max(depth.values(),default=0)-3)*expit(5*late-1.5)),
 'o_delayed_revival':float(expit(8*late-1)*expit(2-6*e10)),
 'o_temporal_concentration':float(expit(6*e25-1.5)*expit(nn-6)),
 'o_branching_persistence':float(expit(root_degree-3)*expit(max(depth.values(),default=0)-2)*expit(4*late-1.2))
 }
 if any(not np.isfinite(x) for x in vals.values()):raise ValueError('Nonfinite harmonized features')
 return vals,{'valid_nodes':nn,'valid_edges':len(edges),'raw_nodes':len(node_t),'raw_edges':len(edge_list),**issues}


def build_source_frame(frame,base):
 rows=[];qa=[]
 for row in frame.itertuples(index=False):
  nd,ed,root,issues=source_tree(base/row.dataset/'tree'/f'{row.id}.txt',row.id)
  feat,counts=features_for_tree(nd,ed,root)
  rows.append({'dataset':row.dataset,'id':row.id,'label':int(row.label),'source_text':row.source_text,'normalized':row.normalized,**feat})
  qa.append({'dataset':row.dataset,'id':row.id,**counts,**{f'parsed_{k}':v for k,v in issues.items()}})
 return pd.DataFrame(rows),pd.DataFrame(qa)


def build_target_frame(frame,graph):
 rows=[];qa=[]
 for row in frame.itertuples(index=False):
  nd,ed,root,issues=graph[row.id]
  feat,counts=features_for_tree(nd,ed,root)
  rows.append({'dataset':row.dataset,'id':row.id,'label':int(row.label),'source_text':row.source_text,'normalized':row.normalized,**feat})
  qa.append({'dataset':row.dataset,'id':row.id,**counts,**issues})
 return pd.DataFrame(rows),pd.DataFrame(qa)


def source_deduplicate(df):
    # Deduplicate source rows without using the external target.
    df=df.sort_values(['dataset','id']).reset_index(drop=True)
    ids={};texts={};keep=[];drops=[]
    for i,r in df.iterrows():
        reasons=[]
        for label,book,key in [('source_id',ids,r.id),('exact_normalized_text',texts,r.normalized)]:
            if key in book:
                reasons.append(label)
                if int(book[key])!=int(r.label):
                    raise RuntimeError('Source duplicate has conflicting labels: '+str(key)[:120])
        if reasons:
            drops.append({'id':r.id,'dataset':r.dataset,'reason':'|'.join(reasons),'normalized_preview':r.normalized[:140]})
        else:
            keep.append(i);ids[r.id]=int(r.label);texts[r.normalized]=int(r.label)
    return df.iloc[keep].reset_index(drop=True),pd.DataFrame(drops)


def target_overlap_audit(source,target,threshold=.93):
 """Source-fitted char TF-IDF for near duplicates; no target fitting or label use."""
 source_ids=set(source.id.astype(str));source_txt=set(source.normalized.astype(str))
 labelfree=target[['id','source_text','normalized']].copy()
 reports=[]
 vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=70000,sublinear_tf=True)
 mat=vec.fit_transform(source.normalized.fillna('').tolist());mat_t=vec.transform(labelfree.normalized.fillna('').tolist())
 for i,r in enumerate(labelfree.itertuples(index=False)):
  reasons=[]
  if str(r.id) in source_ids:reasons.append('same_source_id')
  if r.normalized in source_txt:reasons.append('exact_normalized_text')
  sims=(mat_t[i] @ mat.T).toarray().ravel()
  sim=float(sims.max()) if len(sims) else 0.
  if len(r.normalized)>=22 and sim>=threshold:reasons.append('source_fitted_char_near_duplicate')
  reports.append({'id':str(r.id),'reason':'|'.join(reasons),'max_train_similarity':sim,'exclude':bool(reasons)})
 audit=pd.DataFrame(reports)
 hold=target.loc[~target.id.astype(str).isin(audit.loc[audit.exclude,'id'])].copy().reset_index(drop=True)
 if len(hold)<.70*len(target):raise RuntimeError(f'Excessive external overlap removal: {len(hold)}/{len(target)}')
 return hold,audit


def optional_pheme_overlap(target, input_root:Path, threshold=.93):
    """Optional cross-corpus leakage audit against the previously used PHEME.
    No PHEME labels are read. If PHEME is not provided, do NOT claim this audit passed.
    """
    cands=list(input_root.rglob('pheme_clean.csv')) if input_root.exists() else []
    if not cands:
        return target, pd.DataFrame(columns=['id','pheme_overlap_reason','pheme_max_similarity']), {'performed':False,'reason':'PHEME pheme_clean.csv not attached to Kaggle Input'}
    df=pd.read_csv(cands[0],usecols=['post_id','thread_id','text'],dtype=str,low_memory=False)
    roots=df.loc[df.post_id==df.thread_id,['post_id','text']].drop_duplicates('post_id')
    roots['normal']=roots.text.fillna('').map(norm_text)
    rootids=set(roots.post_id);texts=set(roots.normal)
    vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=70000,sublinear_tf=True)
    mat=vec.fit_transform(roots.normal.tolist());q=vec.transform(target.normalized.tolist())
    reports=[]
    for i,r in enumerate(target.itertuples(index=False)):
        sim=float((q[i]@mat.T).max())
        reason=[]
        if str(r.id) in rootids:reason.append('same_PHEME_source_id')
        if r.normalized in texts:reason.append('same_PHEME_normalized_text')
        if len(r.normalized)>=22 and sim>=threshold:reason.append('PHEME_near_duplicate')
        reports.append({'id':str(r.id),'pheme_overlap_reason':'|'.join(reason),'pheme_max_similarity':sim})
    audit=pd.DataFrame(reports)
    ex=set(audit.loc[audit.pheme_overlap_reason!='','id'])
    clean=target.loc[~target.id.astype(str).isin(ex)].copy().reset_index(drop=True)
    return clean,audit,{'performed':True,'reference':str(cands[0]),'reference_roots':len(roots),'target_excluded':len(ex)}


def make_matrix(df,text,col):
 structural=df[col].to_numpy(dtype=np.float32) if col else np.zeros((len(df),0),dtype=np.float32)
 return np.hstack([text,structural])


def fit_model(train,test,xt,xv,cols,seed=42,placebo_seed=None,return_model=False):
 x=make_matrix(train,xt,cols);z=make_matrix(test,xv,cols)
 if placebo_seed is not None:
  # Entire rule vectors are shuffled within train-only / target-only log-size bins.
  # Keep shared graph/raw evidence intact; retrain with the randomized rules.
  k=len(O);start=xt.shape[1]+len(cols)-k
  for data,xx,offset in [(train,x,0),(test,z,12345)]:
   rg=np.random.default_rng(placebo_seed+offset)
   ns=np.round(data.g_log_nodes.to_numpy(dtype=float)/.75).astype(int)
   for b in np.unique(ns):
    ix=np.where(ns==b)[0]
    if len(ix)>1:
     xx[ix,start:start+k]=xx[rg.permutation(ix),start:start+k].copy()
  if len(cols)<len(O) or cols[-len(O):]!=O:raise ValueError('Placebo only for ontology-extended model')
 model=make_predictor(seed)
 model.fit(x,train.label.to_numpy(dtype=int))
 p=model.predict_proba(z)[:,1]
 return (p,model) if return_model else p


def calc_metrics(y,p,thr):
 y=np.asarray(y,dtype=int);p=np.asarray(p,float);pred=(p>=thr).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
  'mcc':float(matthews_corrcoef(y,pred)), 'recall_rumor':float(recall_score(y,pred,zero_division=0)),
  'precision_rumor':float(precision_score(y,pred,zero_division=0)),
  'accuracy':float(accuracy_score(y,pred)),
  'auprc':float(average_precision_score(y,p)) if len(set(y))>1 else np.nan,
  'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def source_only_oof(df,folds=3,seed=SEED):
 y=df.label.to_numpy(dtype=int);groups=groupids(df)
 cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
 probs={k:np.zeros(len(df)) for k in MODELS};seen=np.zeros(len(df),int)
 fold_log=[]
 for j,(it,iv) in enumerate(cv.split(df,y,groups)):
  tr=df.iloc[it];va=df.iloc[iv]
  xt,xv,_=build_text_encoder(tr.source_text,va.source_text,dim=45)
  for k,cols in MODELS.items():
   probs[k][iv]=fit_model(tr,va,xt,xv,cols,seed+j)
  seen[iv]+=1
  fold_log.append({'fold':j,'n_train':len(tr),'n_valid':len(va),'n_groups_train':len(set(groups[it])),'n_groups_valid':len(set(groups[iv]))})
 if (seen!=1).any():raise RuntimeError('OOF missing/duplicate predictions')
 threshold={k:select_threshold(y,p) for k,p in probs.items()}
 scores={k:calc_metrics(y,p,threshold[k])['macro_f1'] for k,p in probs.items()}
 # Pre-registered complexity penalty in model selection; D is never selected from target.
 penalized={k:v-(.002 if k=='D' else 0.) for k,v in scores.items()}
 selected=max(sorted(penalized),key=lambda k:penalized[k])
 audit=pd.DataFrame([{'model':k,'oof_macro_f1':scores[k],'threshold':threshold[k],'selection_score':penalized[k],'selected':k==selected} for k in MODELS])
 return threshold,selected,audit,pd.DataFrame(fold_log)


def bootstrap_ci(y,pd,pc,t,n=2000,seed=42):
 rng=np.random.default_rng(seed);y=np.asarray(y,dtype=int);pd=np.asarray(pd);pc=np.asarray(pc)
 v=[]
 for _ in range(n):
  ix=rng.integers(0,len(y),len(y))
  v.append(f1_score(y[ix],pd[ix]>=t,average='macro',zero_division=0)-f1_score(y[ix],pc[ix]>=t,average='macro',zero_division=0))
 return float(np.quantile(v,.025)),float(np.quantile(v,.975))


def rdf_provenance(frame,pred,out,max_rows=30):
 from rdflib import Graph,Namespace,URIRef,Literal,RDF,XSD
 g=Graph();EX=Namespace('https://example.org/rumor-validation/');g.bind('ex',EX)
 for i,r in enumerate(frame.head(max_rows).itertuples(index=False)):
  predrow=pred[pred.id==r.id].iloc[0]
  event=EX['event/'+str(i)];ass=EX['prediction/'+str(i)];g.add((ass,RDF.type,EX.ExternalPrediction));g.add((ass,EX.forEvent,event));g.add((event,EX.observedSourceText,Literal(str(r.source_text)[:500])))
  g.add((ass,EX.predictedRumorProbability,Literal(float(predrow.p_D),datatype=XSD.double)))
  for k,rule in enumerate(O):
   evid=EX[f'evidence/{i}/{k}'];g.add((evid,RDF.type,EX.StructuralRuleEvidence));g.add((ass,EX.hasEvidence,evid));g.add((evid,EX.ruleName,Literal(rule)));g.add((evid,EX.ruleScore,Literal(float(getattr(r,rule)),datatype=XSD.double)))
 f=out/'ontology_provenance.ttl';g.serialize(f,format='turtle');verify=Graph().parse(f,format='turtle');return {'triple_count':len(g),'parse_ok':len(verify)==len(g),'sample_count':min(len(frame),max_rows),'warning':'Structural provenance only; no human expert review, OWL entailment, or causal proof'}


def run_evaluation(source,target,out,placebos=49,boots=2000,seed=SEED):
 out.mkdir(parents=True,exist_ok=True)
 thresholds,selected,oof,fold_log=source_only_oof(source,seed=seed)
 oof.to_csv(out/'source_only_oof_model_selection.csv',index=False);fold_log.to_csv(out/'source_only_cv_folds.csv',index=False)
 xt,xv,_=build_text_encoder(source.source_text,target.source_text,dim=45)
 model_probs={};model_objects={}
 for k,col in MODELS.items():
  p,m=fit_model(source,target,xt,xv,col,seed=seed,return_model=True)
  model_probs[k]=p;model_objects[k]=m
 y=target.label.to_numpy(dtype=int)
 t=thresholds['C']
 scores=[]
 for k,p in model_probs.items():
  r=calc_metrics(y,p,thresholds[k]);scores.append({'model':k,'decision_threshold':thresholds[k],'threshold_source':'source_only_oof',**r})
 for k,p in [('C_shared',model_probs['C']),('D_shared',model_probs['D'])]:
  r=calc_metrics(y,p,t);scores.append({'model':k,'decision_threshold':t,'threshold_source':'source_only_C',**r})
 scores.append({'model':'G_source_selected_'+selected,'decision_threshold':thresholds[selected], 'threshold_source':'source_only_selection',**calc_metrics(y,model_probs[selected],thresholds[selected])})
 summary=pd.DataFrame(scores);summary.to_csv(out/'external_summary_metrics.csv',index=False)
 pred=target[['id','label','dataset']].copy()
 for k,p in model_probs.items():pred['p_'+k]=p
 for k in model_probs:pred['pred_'+k]=(pred['p_'+k]>=thresholds[k]).astype(int)
 pred['pred_D_shared']=(pred.p_D>=t).astype(int);pred['pred_C_shared']=(pred.p_C>=t).astype(int)
 pred.to_csv(out/'external_predictions.csv',index=False)
 # Model-fidelity audit: replace each ontology score by its SOURCE median.
 # This is a sensitivity diagnostic, NOT a causal or expert-validity score.
 fidelity=[];base_mat=make_matrix(target,xv,MODELS['D'])
 imps=model_objects['D'].get_feature_importance() if hasattr(model_objects['D'],'get_feature_importance') else []
 if len(imps):
  feature_names=[f'text_svd_{i}' for i in range(xt.shape[1])]+MODELS['D']
  pd.DataFrame({'feature':feature_names,'importance':imps}).to_csv(out/'model_D_feature_importance.csv',index=False)
 for rule in O:
  matrix=base_mat.copy()
  column=xt.shape[1]+MODELS['D'].index(rule)
  matrix[:,column]=float(source[rule].median())
  p_alt=model_objects['D'].predict_proba(matrix)[:,1]
  fidelity.append({'rule':rule,'source_median':float(source[rule].median()),
   'mean_abs_probability_change':float(np.mean(np.abs(p_alt-model_probs['D']))),
   'changed_decisions_shared_C_threshold':int(np.sum((p_alt>=t)!=(model_probs['D']>=t))),
   'interpretation':'single-feature perturbation, not causal proof'})
 pd.DataFrame(fidelity).to_csv(out/'ontology_model_fidelity.csv',index=False)
 lo,hi=bootstrap_ci(y,model_probs['D'],model_probs['C'],t,n=boots,seed=seed)
 delta=calc_metrics(y,model_probs['D'],t)['macro_f1']-calc_metrics(y,model_probs['C'],t)['macro_f1']
 pd.DataFrame([{'comparison':'D_minus_C_shared_C_threshold','delta_macro_f1':delta,'ci95_low':lo,'ci95_high':hi,'test_n':len(y),'bootstrap_repeats':boots}]).to_csv(out/'paired_bootstrap.csv',index=False)
 null=[]
 for repeat in range(placebos):
  p=fit_model(source,target,xt,xv,MODELS['D'],seed=seed,placebo_seed=seed+10000+repeat)
  m=calc_metrics(y,p,t)
  null.append({'repeat':repeat,'macro_f1':m['macro_f1'],'delta_vs_C':m['macro_f1']-calc_metrics(y,model_probs['C'],t)['macro_f1']})
  if (repeat+1)%5==0:
   pd.DataFrame(null).to_csv(out/'ontology_placebo_partial_checkpoint.csv',index=False)
   print('placebo checkpoint',repeat+1,'/',placebos,flush=True)
 null_df=pd.DataFrame(null);null_df.to_csv(out/'ontology_placebo_retrained.csv',index=False)
 real=float(calc_metrics(y,model_probs['D'],t)['macro_f1'])
 placebo_p=(1+int((null_df.macro_f1>=real).sum()))/(1+placebos)
 place_summary={'real_ontology_f1':real,'null_mean_f1':float(null_df.macro_f1.mean()),'null_std':float(null_df.macro_f1.std(ddof=1)) if placebos>1 else 0.,'one_sided_rank_p':float(placebo_p),'repeats':placebos}
 (out/'placebo_summary.json').write_text(json.dumps(place_summary,indent=2),encoding='utf-8')
 rules=[]
 for o in O:
  x=target[o].to_numpy(dtype=float)
  rules.append({'rule':o,'coverage_gt_0_5':float((x>.5).mean()),'saturation_ge_0_95':float((x>=.95).mean()),'std':float(x.std()),'mean':float(x.mean()),'max_abs_corr_with_raw':float(max(abs(np.corrcoef(x,target[r])[0,1]) for r in R if x.std()>1e-8 and target[r].std()>1e-8) if x.std()>1e-8 else 0.)})
 pd.DataFrame(rules).to_csv(out/'ontology_rule_coverage_and_redundancy.csv',index=False)
 rdf_status=rdf_provenance(target,pred,out)
 (out/'rdf_validation.json').write_text(json.dumps(rdf_status,indent=2),encoding='utf-8')
 return {'summary':scores,'delta_D_minus_C':delta,'ci':[lo,hi],'placebo':place_summary,'selected_source_model':selected,'target_n':len(y),'rdf':rdf_status}


def run_pipeline(args):
 try:
  import catboost
 except ImportError as exc:
  raise RuntimeError('CatBoost is required for the frozen external evaluation; install it before running') from exc
 st=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
 base=prepare_data(cache/'twitter_sources',Path(args.input_root) if args.input_root else None,not args.no_download)
 src_labels,inventory=load_datasets(base,strict=True)
 rawsrc,srcqa=build_source_frame(src_labels,base)
 src,dups=source_deduplicate(rawsrc)
 inventory.to_csv(out/'source_inventory.csv',index=False);srcqa.to_csv(out/'source_tree_integrity.csv',index=False);dups.to_csv(out/'source_removed_duplicates.csv',index=False)
 archive=cache/'Twitter-COVID19.7z'
 if not archive.exists():
  if args.no_download:raise RuntimeError('COVID archive is missing; enable Internet')
  download_archive(archive)
 if archive.stat().st_size<1000000:raise RuntimeError('COVID archive unexpectedly small')
 digest=sha256(archive)
 if digest!=EXPECTED_ARCHIVE_SHA:raise RuntimeError(f'Pinned COVID archive hash mismatch {digest}, expected {EXPECTED_ARCHIVE_SHA}. Refuse to score altered data.')
 targetfolder=cache/'twitter_covid_extracted'
 if not list(targetfolder.rglob('Twitter_data_all.txt')):archive_extract(archive,targetfolder)
 rawtarget,graphs,tqa,datapath,labelpath=parse_covid_files(targetfolder)
 target,qa=build_target_frame(rawtarget,graphs)
 valid,audit=target_overlap_audit(src,target)
 audit.to_csv(out/'external_overlap_audit.csv',index=False)
 valid,pheme_audit,pheme_status=optional_pheme_overlap(valid,Path(args.input_root))
 pheme_audit.to_csv(out/'pheme_overlap_audit.csv',index=False)
 (out/'pheme_overlap_status.json').write_text(json.dumps(pheme_status,indent=2),encoding='utf-8')
 qa.to_csv(out/'target_tree_integrity.csv',index=False)
 (out/'target_parse_issues.json').write_text(json.dumps(tqa,indent=2),encoding='utf-8')
 run_info={'source_original_n':len(src_labels),'source_training_n':len(src),'target_total_n':len(target),'target_after_overlap_n':len(valid),'target_overlap_removed_n':len(target)-len(valid),'pheme_overlap_status':pheme_status,'source_repository_sha':'6a33ebd91d4cafeb5a42eabc5727226fec7bb0e5','source_text_mirror_sha':'fc2d35e67d795feab30d22705ae3395eb97b9445','covid_repo_sha':COVID_SHA,'covid_archive_sha256':digest,'time_delay_unit':'NOT_VERIFIED','primary_window':'full','temporal_features':'scale-invariant within-tree relative delay','feature_lists':{'graph':G,'raw':R,'ontology':O},'model_family':'CatBoost if installed else HistGradientBoosting','target_labels_for_tuning':False,'cv_groups':'source-only exact and near duplicate','external_data_previous_status':'preflight only','design_disclosure':'Harmonized source-retrained protocol, not original PHEME/T15/16 model weights','seed':args.seed,'placebos':args.placebos,'bootstrap':args.bootstrap,'catboost_version':catboost.__version__,'pipeline_sha256':simple_file_sha(Path(__file__))}
 (out/'FROZEN_EXTERNAL_PROTOCOL.json').write_text(json.dumps(run_info,indent=2,ensure_ascii=False),encoding='utf-8')
 r=run_evaluation(src,valid,out,placebos=args.placebos,boots=args.bootstrap,seed=args.seed)
 r['run_info']=run_info;r['run_seconds']=round(time.time()-st,2)
 (out/'RESULTS_SUMMARY.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8')
 md=['# Twitter-COVID19 external validation — source-only harmonized frozen protocol',
     '\nPrimary comparison: D (ontology) vs C (raw evidence), **same C threshold from source-only OOF**.',
     'No target labels used for feature/threshold/model selection. Report target labels only for final metrics.',
     'COVID time-delay units unverified; **no claim of 1h/6h/24h early detection**.',
     '\n## Counts',f"- Source before/after source-only deduplication: {len(src_labels)} / {len(src)}",
     f"- Target before/after overlap exclusion: {len(target)} / {len(valid)}",
     '\n## Summary',pd.DataFrame(r['summary']).to_markdown(index=False),
     f"\nD−C Macro-F1 = {r['delta_D_minus_C']:.4f}, 95% paired bootstrap CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
     f"Conditional placebo mean = {r['placebo']['null_mean_f1']:.4f}, Monte Carlo p = {r['placebo']['one_sided_rank_p']:.3f}",
     '\n## Critical limitations',
     '- Structural fuzzy rules are deterministic transformations of graphs. They are not ground truth verification or expert-validated explanations.',
     '- COVID19 lacks user identities in the available files: no author-diversity rule is used.',
     '- The harmonized architecture was frozen before target scoring, but the prior Twitter15/16 development history and target domain choice should be disclosed.',
     '- Target time delays are relative and units not independently verified. Full-tree is the only confirmatory window.',
     '- Placebo conditional shuffling assesses predictive association under a null; it is not a randomized causal experiment.']
 (out/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
 (out/'run_state.json').write_text(json.dumps({'completed':True,'target_n':len(valid)},indent=2))
 bundle=out.parent/(out.name+'.zip')
 with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(out))
 print('COMPLETE: '+str(bundle));return bundle


def selftest():
 # Test graph root, reachability, temporal ordering, and ID-independent rule computation.
 nodes={'1':0.,'2':4.,'3':2.,'4':-1.,'5':7.}
 ed=[('1','2'),('1','3'),('2','5'),('3','4'),('5','1'),('1','2')]
 f,qa=features_for_tree(nodes,ed,'1')
 assert qa['valid_nodes']==4 and qa['valid_edges']==3,(f,qa)
 assert qa.get('backward_edge',0)>0 and qa.get('missing_or_negative_node',0)>0
 assert len(f)==len(G)+len(R)+len(O) and all(np.isfinite(list(f.values())))
 assert len(O)==4 and not any('user' in x or 'independent' in x for x in G+R+O)
 print('Parser/structural/ontology invariant self-tests PASS')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='/kaggle/working/TwitterCOVID19_Frozen_External_Results');ap.add_argument('--cache',default='/kaggle/working/twitter_external_cache');ap.add_argument('--input-root',default='/kaggle/input');ap.add_argument('--no-download',action='store_true');ap.add_argument('--placebos',type=int,default=49);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 if args.self_test:selftest()
 else:run_pipeline(args)
