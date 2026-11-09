"""PHEME v13: Event-held-out, explainable rumor detection with typed ontology evidence.

Scientific guardrails:
- Frozen generic MiniLM, no target-label features or users' label-based priors.
- Every trained vectorizer, scaler, PCA, classifier, and threshold is fit/tuned on
  the OUTER TRAINING EVENTS ONLY. Inner folds hold out one training event each.
- Explicit **trained** branch ablations C/D/E/F/G, plus forced-positive diagnostics.
- Retrospective full-thread setting: replies observed after posting are permitted.
- Ontology rules are deterministic operational evidence, not OWL-DL truth inference.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score, matthews_corrcoef
import pheme_v11_features as old
from pheme_v14_rules import RULES14, DESCRIPTIONS14, build_v14, assert_label_blind

EV_RULES = [
 'o12_direct_refutation','o12_independent_refutation','o12_cited_refutation',
 'o12_direct_verification_request','o12_independent_verification','o12_independent_support',
 'o12_cross_author_dispute','o12_early_refutation','o12_parent_chain_dispute',
 'o12_citation_dispute','o12_support_without_verification','o12_source_uncertainty_dispute',
 'o12_author_diversity_strength','o12_evidence_entropy','o12_direct_evidence_strength',
 'o12_early_evidence_strength','o12_contradiction_confidence','o12_unverified_claim_evidence',
 'o12_independent_evidence_strength','o12_credible_support_strength',
 'o12_multi_path_conflict','o12_claim_evidence_uncertainty']

PAT = {
 'denial': re.compile(r"\b(?:fake|hoax|false|untrue|fabricat(?:ed|ion)|debunk(?:ed|ing)?|misleading|not true|isn.t true|incorrect|not accurate|no evidence|no proof|made up|this is a lie|rumou?rs?)\b",re.I),
 'query': re.compile(r"\b(?:is (?:this|that|it) (?:true|real|confirmed)|any (?:proof|evidence|source)|where.s (?:the|your) (?:source|proof)|can (?:anyone|somebody) (?:confirm|verify)|has (?:this|it) been confirmed|how (?:do|can) (?:you|we) know|source\?|verified\?|proof\?)",re.I),
 'support': re.compile(r"\b(?:confirmed|verified|officially confirmed|it.s true|this is true|indeed true|accurate report|can confirm|yes,? (?:it is|that is) true)\b",re.I),
 'uncertainty': re.compile(r"\b(?:unconfirmed|unverified|reportedly|allegedly|apparently|might be|may be|not sure|unclear|rumou?rs?|seems like|could be)\b",re.I),
 'citation': re.compile(r"(?:https?://\S+|www\.\S+|\b(?:reuters|ap news|bbc|associated press|official statement|according to)\b)",re.I),
}

SOURCE_FIELDS = ['source_text','thread_id','event_id','label']
CONFIGS = [
 'A_WordChar_Source',
 'B_Text_Reply_MiniLM',
 'C_Interaction_Semantic',
 'D_Interaction_Graph',
 'E_Interaction_RawEvidence',
 'F_Interaction_OntologyV11',
 'G_Interaction_OntologyV12',
 'H_Interaction_OntologyV13',
 'I_StabilityPruned_Ontology',
 'J_EventCV_Selected',
 'K_Interaction_OntologyV14',
 'L_Stable_OntologyV14',
 'M_CatBoost_OntologyV14',
 'N_EventCV_Selected_v14',
]




# v13 Evidence Quality: continuous, label-blind operational rules.
# These scores are hypotheses based on observed conversation signals, NOT
# logically entailed truth/falsity, nor the stance of a verified fact-checker.
O13_RULES = [
    'o13_evidence_density','o13_discussion_uncertainty','o13_author_independence',
    'o13_consensus_tension','o13_semantic_novelty_evidence','o13_reply_topology_pressure',
    'o13_provenance_diversity','o13_evidence_coverage','o13_temporal_evolution',
    'o13_claim_dispute_reliability', 'o13_direct_argument_quality',
    'o13_weak_evidence_penalty'
]
O13_DESCRIPTIONS = [
    'Observed lexical evidence fraction, weighted by sampled reply coverage',
    'Uncertainty and evidence entropy weighted by breadth of discussion',
    'Unique-author share, corrected for size of the sampled conversation',
    'Support versus challenge imbalance in sampled replies',
    'Semantic novelty weighted by lexical uncertainty evidence',
    'Depth and breadth signal interacting with evidence disagreement',
    'Independence of challenged or cited claims across observed authors',
    'Effective sample coverage of the reply graph',
    'Variation in early and later reply semantics',
    'Source uncertainty interacting with reply challenges and citation absence',
    'Direct reply evidence with citation and independent authorship',
    'Uncorroborated uncertainty fraction without source citation cues'
]

def ontology_v13(data):
    """Label-blind evidence-quality interactions; values bounded in [0, 1]."""
    def x(name):
        z=np.asarray(data[name],dtype=np.float32)
        return np.nan_to_num(z,nan=0,posinf=0,neginf=0)
    def unit(v):return np.clip(v,0,1)
    coverage=unit(x('e_reply_coverage'))
    diverse=unit(x('unique_users')/np.maximum(x('num_nodes'),1))
    breadth=unit(np.log1p(x('num_nodes'))/np.log1p(200))
    depth=unit(x('thread_depth')/15)
    challenge=unit(x('e_challenge_ratio'))
    query=unit(x('e_query_rate'))
    denial=unit(x('e_deny_rate'))
    support=unit(x('e_support_rate'))
    citation=unit(x('e_citation_rate'))
    uncertainty=unit(x('e_uncertainty_rate'))
    entropy=unit(x('o12_evidence_entropy'))
    novelty=unit(x('e_semantic_novelty_rate'))
    semvar=unit(x('e_semantic_sim_std'))
    drift=unit(np.abs(x('e_semantic_early_late_shift')))
    source_unc=unit(x('e_source_uncertainty'))
    source_cit=unit(x('e_source_citation'))
    authors=unit(x('e_dispute_diversity'))
    direct=unit(x('o12_direct_evidence_strength'))
    cited=unit(x('e_cited_denial_rate')+x('e_cited_support_rate'))
    scores={
      'o13_evidence_density':unit((.45*challenge+.30*uncertainty+.25*citation)*(.35+.65*coverage)),
      'o13_discussion_uncertainty':unit((.50*entropy+.35*uncertainty+.15*source_unc)*(.25+.75*breadth)),
      'o13_author_independence':unit(diverse*(.35+.65*breadth)),
      'o13_consensus_tension':unit(np.abs(support-challenge)*(.30+.70*entropy)),
      'o13_semantic_novelty_evidence':unit((.70*novelty+.30*semvar)*(.35+.65*challenge)),
      'o13_reply_topology_pressure':unit((.6*depth+.4*breadth)*(.35+.65*entropy)),
      'o13_provenance_diversity':unit((.50*authors+.50*diverse)*(.45+.55*citation)),
      'o13_evidence_coverage':unit(coverage*(.35+.65*breadth)),
      'o13_temporal_evolution':unit((.7*drift+.3*semvar)*(.25+.75*breadth)),
      'o13_claim_dispute_reliability':unit((.5*query+.5*denial)*(.45+.55*source_unc)*(1-.5*cited)),
      'o13_direct_argument_quality':unit((.5*direct+.5*cited)*(.35+.65*diverse)),
      'o13_weak_evidence_penalty':unit((.6*source_unc+.4*uncertainty)*(1-.5*source_cit)*(1-.5*citation)),
    }
    result=pd.DataFrame(scores,index=data.index,dtype='float32')
    assert list(result.columns)==O13_RULES
    assert np.isfinite(result.to_numpy()).all()
    assert (result.to_numpy()>=0).all() and (result.to_numpy()<=1).all()
    return result

# Applied to metadata features *inside* each inner/outer training fold. Neither
# held-out labels nor held-out covariate distributions choose the feature mask.
CORE_KEEP = {'t_source_length','t_source_wordcount','t_source_qmarks',
             't_source_hashtagcount','t_source_links','t_source_exclamation',
             'num_nodes','thread_depth','reply_time_span_log','reply_time_median_log',
             'e_n_replies_log','e_reply_coverage','e_semantic_sim_mean',
             'e_semantic_sim_std','e_semantic_available'}

def train_only_relevance_filter(frame,columns,train,events):
    kept=[]; dropped=[]
    ev=np.asarray(events);train=np.asarray(train)
    for c in columns:
        v=frame.iloc[train][c].to_numpy(dtype=float)
        if c in CORE_KEEP:
            kept.append(c);continue
        if np.unique(v).size<2:
            dropped.append((c,'constant_on_training_events'));continue
        ne=0
        for e in np.unique(ev[train]):
            z=v[ev[train]==e]
            if np.count_nonzero(np.abs(z)>1.e-10)>=max(4,int(np.ceil(.01*len(z)))):
                ne+=1
        if ne>=2:kept.append(c)
        else:dropped.append((c,'rare_or_single_event'))
    if not kept:raise ValueError('Training-only filter removed all features')
    return kept,dropped

def ontology_v12(df:pd.DataFrame, replies:pd.DataFrame):
    """Materialize typed thread evidence using own conversation only, target blind.

    Outputs: 22 continuous rule activations with valid [0,1] bounds and audit rows.
    A soft rule is not an independent ground-truth fact or proof of falsity.
    """
    grouped={str(k):g for k,g in replies.groupby('thread_id',sort=False)}
    rows=[];audit=[]
    for th,su,src in df[['thread_id','source_user','source_text']].itertuples(index=False,name=None):
        g=grouped.get(str(th))
        d=dict.fromkeys(EV_RULES,0.)
        if g is None or len(g)==0:
            rows.append(d);continue
        n=len(g)
        m={q:g[q].to_numpy(dtype=float) for q in ('denial','query','support','citation','uncertainty')}
        direct=g['direct'].to_numpy(dtype=bool)
        users=g['user_id'].fillna('').astype(str).tolist()
        par=g['reply_to'].fillna('').astype(str).tolist()
        ids=g['post_id'].fillna('').astype(str).tolist()
        id2idx={pid:i for i,pid in enumerate(ids)}
        root=str(th)
        # parent relations use exact, string-preserved Twitter IDs
        parent_reply=np.array([(p in id2idx and p!=pid) for pid,p in zip(ids,par)])
        distinct_users=set(x for x in users if x)
        valid_n=max(1,len(distinct_users))
        denied={u for u,k in zip(users,m['denial']) if u and k>0}
        queried={u for u,k in zip(users,m['query']) if u and k>0}
        supported={u for u,k in zip(users,m['support']) if u and k>0}
        challenged=denied|queried
        # citation is a trace of a citation cue, not a verified external source
        denial=m['denial'];query=m['query'];support=m['support'];cite=m['citation']
        observed=min(8,n)
        k=(denial+query>0).astype(float)
        rate=lambda x: float(np.mean(x)) if len(x) else 0.
        d['o12_direct_refutation']=rate(denial[direct]) if direct.any() else 0.
        d['o12_independent_refutation']=min(1.,len(denied)/max(2.,math.sqrt(valid_n)))
        d['o12_cited_refutation']=rate((denial*cite))
        d['o12_direct_verification_request']=rate(query[direct]) if direct.any() else 0.
        d['o12_independent_verification']=min(1.,len(queried)/max(2.,math.sqrt(valid_n)))
        d['o12_independent_support']=min(1.,len(supported)/max(2.,math.sqrt(valid_n)))
        d['o12_cross_author_dispute']=float(bool(denied and supported and any(a!=b for a in denied for b in supported)))
        d['o12_early_refutation']=rate(denial[:min(3,observed)])
        # Reply-to-reply path: a denial cue follows an earlier reply to this thread
        d['o12_parent_chain_dispute']=rate((parent_reply*denial))
        d['o12_citation_dispute']=rate(cite*k)
        d['o12_support_without_verification']=rate(support*(1-cite))
        d['o12_source_uncertainty_dispute']=float(bool(PAT['uncertainty'].search(str(src))))*rate(k)
        d['o12_author_diversity_strength']=float(min(1.,len(distinct_users)/max(n,1)))
        probs=np.array([np.sum(denial),np.sum(query),np.sum(support),max(0.,n-np.sum(k)-np.sum(support))],dtype=float)
        probs=probs/(probs.sum()+1.e-12)
        d['o12_evidence_entropy']=float(max(0.,-np.sum(np.where(probs>0,probs*np.log(np.maximum(probs,1e-12)),0.))/math.log(4)))
        d['o12_direct_evidence_strength']=np.clip(.55*d['o12_direct_refutation']+.45*d['o12_direct_verification_request'],0,1)
        d['o12_early_evidence_strength']=rate(k[:min(3,observed)])
        d['o12_contradiction_confidence']=np.sqrt(rate(denial)*rate(support)) * d['o12_author_diversity_strength']
        d['o12_unverified_claim_evidence']=rate(query)*(1-rate(cite*support))
        d['o12_independent_evidence_strength']=min(1.,len(challenged)/max(2.,math.sqrt(valid_n)))*rate(k)
        d['o12_credible_support_strength']=rate(support*cite)
        d['o12_multi_path_conflict']=np.clip(d['o12_parent_chain_dispute']*d['o12_cross_author_dispute']+.5*d['o12_citation_dispute'],0,1)
        d['o12_claim_evidence_uncertainty']=np.clip(.4*rate(query)+.3*rate(denial)+.3*d['o12_source_uncertainty_dispute']-.2*d['o12_credible_support_strength'],0,1)
        for f,v in d.items():
            if not np.isfinite(v) or v < -1e-7 or v > 1+1e-7:
                raise RuntimeError(f'Invalid ontology value {f}: {v}')
        rows.append(d)
        if len(audit)<800:
            fired=[f for f,v in d.items() if v>0]
            if fired:
                audit.append({'thread_id':str(th),'n_sampled_replies':int(n),'source_post_id':str(th),
                              'rule_ids':';'.join(fired),'rule_count':len(fired),
                              'reply_post_ids':';'.join(ids[:8]),'evidence_author_ids':';'.join(sorted(distinct_users)[:8]),
                              'rule_scores_json':json.dumps({k:round(float(v),5) for k,v in d.items() if v>0}),
                              'reply_evidence_json':json.dumps([{'id':str(pid),'parent':str(parent),'author':str(u),
                                  'direct':bool(dr),'deny':int(a),'query':int(b),'support':int(c),'citation':int(cit)}
                                  for pid,parent,u,dr,a,b,c,cit in zip(ids[:8],par[:8],users[:8],direct[:8],
                                  denial[:8],query[:8],support[:8],cite[:8])])})
    result=pd.DataFrame(rows,index=df.index)[EV_RULES].astype('float32')
    if np.isnan(result.to_numpy()).any(): raise RuntimeError('NaN in Ontology v12 features')
    return result, pd.DataFrame(audit)


def check_label_blind(df,reps):
    # Engineered ontology rules must be invariant to label and event perturbations.
    inp=df.copy()
    cols=inp[['label','event_id']].copy()
    a,_=ontology_v12(inp,reps)
    inp['label']=1-inp['label'].astype(int)
    inp['event_id']='CHANGED_TEST_EVENT'
    b,_=ontology_v12(inp,reps)
    np.testing.assert_array_equal(a.to_numpy(),b.to_numpy())
    assert a.shape==(len(df),len(EV_RULES))
    return True


def event_class_weights(y,ev,idx):
    y0=y[idx];e0=ev[idx]; w=np.zeros(len(idx),np.float32)
    unique=np.unique(e0)
    for evn in unique:
        for label in [0,1]:
            mask=(e0==evn)&(y0==label)
            if np.any(mask):w[mask]=1/(2*len(unique)*np.sum(mask))
    return w/max(np.mean(w),1e-9)


def make_texts(df,reps):
    grouped={str(t):g for t,g in reps.groupby('thread_id',sort=False)}
    source=df.source_text.fillna('').astype(str).tolist()
    context=[]
    for t,s in zip(df.thread_id.astype(str),source):
        rr=grouped.get(t)
        # The first three replies are observed historical context, not labels.
        snippets=rr.text.fillna('').astype(str).iloc[:3].tolist() if rr is not None else []
        context.append((s+' '+ ' '.join('[REPLY] '+x[:280] for x in snippets)).strip())
    return source,context


def clf_text(train,test,strings,y,ev,hash_x,seed):
    # Fit TF-IDF vocabulary/IDF on TRAIN ONLY
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=2,max_df=.98,
                        strip_accents='unicode',sublinear_tf=True,dtype=np.float32)
    a=vec.fit_transform([strings[i] for i in train]); b=vec.transform([strings[i] for i in test])
    a=hstack([a,hash_x[train]],format='csr');b=hstack([b,hash_x[test]],format='csr')
    model=LogisticRegression(C=1.,solver='liblinear',max_iter=400,random_state=seed)
    model.fit(a,y[train],sample_weight=event_class_weights(y,ev,train))
    return model.predict_proba(b)[:,1].astype('float32')


def clf_emb(train,test,source_emb,y,ev,seed):
    if source_emb is None:return np.zeros(len(test),np.float32)+.5
    ss=StandardScaler().fit(source_emb[train]);a=np.clip(ss.transform(source_emb[train]),-6,6);b=np.clip(ss.transform(source_emb[test]),-6,6)
    model=LogisticRegression(C=.15,solver='liblinear',max_iter=400,random_state=seed)
    model.fit(a,y[train],sample_weight=event_class_weights(y,ev,train))
    return model.predict_proba(b)[:,1].astype('float32')


def make_dense_sem(train,test,embed):
    if embed is None:return np.empty((len(train),0),np.float32),np.empty((len(test),0),np.float32)
    # PCA fit ONLY on the train split (not on the outer test nor inner validation)
    # frozen embeddings are generic; compression itself remains split-specific
    a=embed[train]; b=embed[test]
    split=int((a.shape[1]-9)/3) if (a.shape[1]-9)%3==0 else None
    if split is None or split<32:raise ValueError('Unexpected source/reply embeddings shape')
    parts=[];parts_b=[]
    for start,stop in ((0,split),(split,2*split),(2*split,3*split)):
        comp=min(20,len(train)-1,stop-start)
        pca=PCA(n_components=comp,svd_solver='randomized',random_state=42)
        parts.append(pca.fit_transform(a[:,start:stop]).astype('float32'))
        parts_b.append(pca.transform(b[:,start:stop]).astype('float32'))
    parts.append(a[:,3*split:].astype('float32'));parts_b.append(b[:,3*split:].astype('float32'))
    return np.column_stack(parts).astype('float32'),np.column_stack(parts_b).astype('float32')


def train_only_projection(train,test,emb,source_strings,context_strings,fast=False):
    """Training-only text+semantic compression for nonlinear evidence interactions."""
    atr,bte=make_dense_sem(train,test,emb)
    pieces_tr=[atr];pieces_te=[bte]
    dim=8 if fast else 16
    for strings in (source_strings,context_strings):
        vec=TfidfVectorizer(ngram_range=(1,2),max_features=16000,min_df=2,
                             sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
        xx=vec.fit_transform([strings[i] for i in train]);
        yy=vec.transform([strings[i] for i in test])
        k=min(dim,xx.shape[1]-1,max(1,len(train)-1))
        if k<1:continue
        svd=TruncatedSVD(n_components=k,n_iter=4 if not fast else 2,random_state=42)
        pieces_tr.append(svd.fit_transform(xx).astype('float32'))
        pieces_te.append(svd.transform(yy).astype('float32'))
    return np.column_stack(pieces_tr).astype('float32'),np.column_stack(pieces_te).astype('float32')

def clf_dense(train,test,frame,columns,emb,y,ev,seed,low_memory=False,
              projection=None, variant='v12'):
    """Fold-local training and optional evidence stability pruning."""
    from lightgbm import LGBMClassifier
    if variant in ('regularized','v14_stable'):
        selected,dropped=train_only_relevance_filter(frame,columns,train,ev)
    else:
        selected=list(columns);dropped=[]
    xtr=frame.iloc[train][selected].to_numpy(dtype=np.float32)
    xte=frame.iloc[test][selected].to_numpy(dtype=np.float32)
    semtr,semte=projection if projection is not None else make_dense_sem(train,test,emb)
    xtr=np.column_stack([semtr,xtr]);xte=np.column_stack([semte,xte])
    if not np.isfinite(xtr).all() or not np.isfinite(xte).all():
        raise ValueError('Nonfinite input in metadata branch')
    if variant in ('regularized','v14_stable'):
        hp=dict(n_estimators=260 if not low_memory else 150,learning_rate=.035,
          num_leaves=7,max_depth=3,min_child_samples=80,
          reg_lambda=15.,reg_alpha=1.,colsample_bytree=.75,
          subsample=.80,subsample_freq=1,verbosity=-1,n_jobs=2,random_state=seed)
    elif variant=='v14':
        hp=dict(n_estimators=300 if not low_memory else 190,learning_rate=.028,
          num_leaves=7,max_depth=3,min_child_samples=65,reg_lambda=17.,
          reg_alpha=1.5,colsample_bytree=.85,subsample=.85,subsample_freq=1,
          verbosity=-1,n_jobs=2,random_state=seed)
    else:
        hp=dict(n_estimators=220 if not low_memory else 130,learning_rate=.04,
          num_leaves=9,max_depth=4,min_child_samples=55,
          reg_lambda=8.,reg_alpha=.35,colsample_bytree=.85,
          subsample=.85,subsample_freq=1,verbosity=-1,n_jobs=2,random_state=seed)
    if variant=='catboost':
        from catboost import CatBoostClassifier
        m=CatBoostClassifier(iterations=250 if not low_memory else 140,depth=4,
          learning_rate=.04,l2_leaf_reg=12.,loss_function='Logloss',
          random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    else:
        m=LGBMClassifier(**hp)
    names=[f'f_{i}' for i in range(xtr.shape[1])]
    m.fit(pd.DataFrame(xtr,columns=names),y[train],sample_weight=event_class_weights(y,ev,train))
    m._v13_selected=selected;m._v13_dropped=dropped
    return m.predict_proba(pd.DataFrame(xte,columns=names))[:,1].astype('float32'),m


def _scores(y,p,events,thresholds):
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,float); events=np.asarray(events)
    event_scores=[]
    for e in np.unique(events):
        m=events==e
        truth=y[m,None].astype(bool); guess=p[m,None]>=thresholds[None,:]
        tp=np.count_nonzero(truth&guess,axis=0).astype(float)
        tn=np.count_nonzero(~truth&~guess,axis=0).astype(float)
        fp=np.count_nonzero(~truth&guess,axis=0).astype(float)
        fn=np.count_nonzero(truth&~guess,axis=0).astype(float)
        fpos=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
        fneg=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)>0)
        event_scores.append(.5*(fpos+fneg))
    arr=np.stack(event_scores)
    # robust cross-event objective; never reads outer test labels
    return .85*np.mean(arr,axis=0)+.15*np.min(arr,axis=0)


def tune(y,p,events):
    grid=np.linspace(.18,.82,65)
    values=_scores(y,p,events,grid)
    i=max(range(len(grid)),key=lambda j:(values[j],-abs(grid[j]-.5)))
    return float(grid[i]),float(values[i])


def metrics(y,p,threshold):
    yy=(np.asarray(p)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yy,labels=[0,1]).ravel()
    denp=2*tp+fp+fn;denn=2*tn+fp+fn
    return dict(MacroF1=float(.5*((2*tp/denp if denp else 0.)+(2*tn/denn if denn else 0.))),
                RumorF1=float(2*tp/denp if denp else 0.),
                Recall=float(tp/(tp+fn) if tp+fn else 0.),
                Precision=float(tp/(tp+fp) if tp+fp else 0.),
                Accuracy=float(np.mean(yy==y)),
                MCC=float(matthews_corrcoef(y,yy)),
                AUPRC=float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan,
                ROC_AUC=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
                TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))


def prepare_all(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data,embed,source,traces,reps=old.prepare(args.pheme,args.graph,out,max_replies=args.max_replies,
        encode=not args.no_minilm,batch_size=args.batch_size,cache_dir=args.cache)
    f12,audit=ontology_v12(data,reps)
    check_label_blind(data,reps)
    data=pd.concat([data,f12],axis=1)
    v13=ontology_v13(data)
    # Explicit permutation test: target labels and event names cannot influence v13 ontology.
    disturbed=data.copy();disturbed['label']=1-disturbed['label'].astype(int)
    disturbed['event_id']='PERTURBED'
    np.testing.assert_array_equal(v13.to_numpy(),ontology_v13(disturbed).to_numpy())
    data=pd.concat([data,v13],axis=1)
    # v14 provenance-aware, independent-author evidence rules; independent of every label.
    assert_label_blind(data,reps)
    v14,audit14=build_v14(data,reps)
    data=pd.concat([data,v14],axis=1)
    v14.to_csv(out/'ontology_v14_label_blind_features.csv',index_label='row')
    pd.DataFrame({'rule':RULES14,'interpretation':DESCRIPTIONS14}).to_csv(out/'ontology_v14_rule_catalog.csv',index=False)
    audit14.to_csv(out/'ontology_v14_provenance.csv',index=False)
    (pd.DataFrame({'rule':O13_RULES,'interpretation':O13_DESCRIPTIONS})
       .to_csv(out/'ontology_v13_rule_catalog.csv',index=False))
    # Label-blind textual morphology: serves as an interaction baseline even without embeddings
    a=data.source_text.fillna('').astype(str)
    data['t_source_length']=np.log1p(a.str.len()).astype('float32')
    data['t_source_wordcount']=np.log1p(a.str.split().str.len()).astype('float32')
    data['t_source_qmarks']=(a.str.count(r'\?')>0).astype('float32')
    data['t_source_hashtagcount']=np.log1p(a.str.count('#')).astype('float32')
    data['t_source_links']=(a.str.contains('http',case=False,regex=False)).astype('float32')
    data['t_source_exclamation']=np.log1p(a.str.count('!')).astype('float32')
    data[['thread_id',*EV_RULES,*O13_RULES]].to_csv(out/'ontology_v13_label_blind_features.csv',index=False)
    audit.to_csv(out/'ontology_v12_evidence_provenance.csv',index=False)
    cov=[]
    for n in old.GRAPH+old.RAW+old.RULES+EV_RULES+O13_RULES+RULES14:
        x=data[n].to_numpy(dtype=float)
        cov.append({'feature':n,'nonzero_fraction':float(np.mean(x!=0)),'n_unique':int(len(np.unique(x)))})
    pd.DataFrame(cov).to_csv(out/'feature_coverage.csv',index=False)
    return data,embed,source,reps


def ontology_rdf(audit,out):
    """RDF schema + provenance examples + SHACL constraints (not OWL-DL reasoning)."""
    try:
        from rdflib import Graph,Namespace,RDF,RDFS,OWL,Literal,URIRef,XSD
    except ImportError:return
    ns=Namespace('https://example.org/pheme/ontology/v12/')
    sh=Namespace('http://www.w3.org/ns/shacl#')
    schema=Graph();schema.bind('ev',ns);schema.bind('owl',OWL)
    schema.add((ns['OntologyV12'],RDF.type,OWL.Ontology))
    for c in ['Claim','Reply','Author','Evidence','RuleActivation']:
        schema.add((ns[c],RDF.type,OWL.Class))
    for prop,domain,rng in [('refersToClaim','Reply','Claim'),('hasReply','Claim','Reply'),
                             ('authoredBy','Reply','Author'),('triggeredRule','RuleActivation','Evidence'),
                             ('aboutClaim','RuleActivation','Claim'),('observedReply','RuleActivation','Reply')]:
        schema.add((ns[prop],RDF.type,OWL.ObjectProperty))
        schema.add((ns[prop],RDFS.domain,ns[domain]));schema.add((ns[prop],RDFS.range,ns[rng]))
    schema.add((ns.confidence,RDF.type,OWL.DatatypeProperty))
    schema.add((ns.confidence,RDFS.range,XSD.decimal))
    for r in EV_RULES:schema.add((ns[r],RDF.type,ns.Evidence))
    schema.serialize(destination=str(out/'ontology_v12_schema.ttl'),format='turtle')
    shapes=Graph();shapes.bind('sh',sh);shapes.bind('ev',ns)
    subj=ns.RuleActivationShape
    shapes.add((subj,RDF.type,sh.NodeShape));shapes.add((subj,sh.targetClass,ns.RuleActivation))
    for prop,datatype,minimum,maximum in [
          (ns.confidence,XSD.decimal,0,1)]:
        node=ns.ConfidencePropertyShape
        shapes.add((subj,sh.property,node));shapes.add((node,sh.path,prop))
        shapes.add((node,sh.datatype,datatype));shapes.add((node,sh.minCount,Literal(1)))
        shapes.add((node,sh.minInclusive,Literal(minimum)));shapes.add((node,sh.maxInclusive,Literal(maximum)))
    cl=ns.ClaimReferenceShape
    shapes.add((subj,sh.property,cl));shapes.add((cl,sh.path,ns.aboutClaim));
    shapes.add((cl,sh.minCount,Literal(1)))
    shapes.serialize(destination=str(out/'ontology_v12_shapes.ttl'),format='turtle')
    g=Graph();g.bind('ev',ns)
    for item in audit.head(30).itertuples(index=False):
        cl=ns['claim_'+str(item.thread_id)];g.add((cl,RDF.type,ns.Claim))
        try: scores=json.loads(item.rule_scores_json)
        except (TypeError,ValueError):scores={}
        for rule,value in scores.items():
            r=ns['activation_'+str(item.thread_id)+'_'+rule]
            g.add((r,RDF.type,ns.RuleActivation));g.add((r,ns.aboutClaim,cl))
            g.add((r,ns.triggeredRule,ns[rule]))
            g.add((r,ns.confidence,Literal(str(round(float(value),5)),datatype=XSD.decimal)))
        try: replies=json.loads(item.reply_evidence_json)
        except (TypeError,ValueError):replies=[]
        for rep in replies[:8]:
            if not rep.get('id'):continue
            r=ns['post_'+str(rep['id'])]
            g.add((r,RDF.type,ns.Reply));g.add((cl,ns.hasReply,r));g.add((r,ns.refersToClaim,cl))
            if rep.get('author'):
                u=ns['user_'+str(rep['author'])];g.add((u,RDF.type,ns.Author));g.add((r,ns.authoredBy,u))
            # Observed evidence is not claimed as human-verified support or refutation.
            for rule in scores:
                a=ns['activation_'+str(item.thread_id)+'_'+rule]
                g.add((a,ns.observedReply,r))
    g.serialize(destination=str(out/'ontology_v12_evidence_sample.ttl'),format='turtle')
    # Verification of generated RDF syntax and ontology/shape readability.
    for f in ('ontology_v12_schema.ttl','ontology_v12_shapes.ttl','ontology_v12_evidence_sample.ttl'):
        assert len(Graph().parse(str(out/f),format='turtle'))>0,f



def ontology_rdf_v14(audit,out):
    from rdflib import Graph, Namespace, RDF, RDFS, XSD, Literal, OWL
    ex=Namespace('https://example.org/pheme/evidence/v14/')
    g=Graph();g.bind('ev14',ex)
    for c in ['Claim','Reply','Author','EvidenceCue','RuleActivation']:
        g.add((ex[c],RDF.type,OWL.Class))
    for rel in ['supportedByReply','authoredBy','aboutClaim','observedIn']:
        g.add((ex[rel],RDF.type,OWL.ObjectProperty))
    g.add((ex.strength,RDF.type,OWL.DatatypeProperty));g.add((ex.strength,RDFS.range,XSD.decimal))
    for row in audit.head(60).itertuples(index=False):
        claim=ex['claim_'+str(row.thread_id)]
        g.add((claim,RDF.type,ex.Claim))
        for val in json.loads(row.reply_evidence_json):
            ident=str(val.get('id',''));author=str(val.get('author',''))
            if not ident:continue
            rr=ex['reply_'+ident]
            g.add((rr,RDF.type,ex.Reply));g.add((claim,ex.supportedByReply,rr))
            if author:
                user=ex['author_'+author]
                g.add((user,RDF.type,ex.Author));g.add((rr,ex.authoredBy,user))
        rule_scores=json.loads(row.rules_json)
        for rule,confidence in rule_scores.items():
            if rule not in RULES14:continue
            act=ex['activation_'+str(row.thread_id)+'_'+rule]
            g.add((act,RDF.type,ex.RuleActivation));g.add((act,ex.aboutClaim,claim))
            g.add((act,ex.observedIn,ex[rule]))
            g.add((act,ex.strength,Literal(str(float(confidence)),datatype=XSD.decimal)))
    path=out/'ontology_v14_provenance_sample.ttl'
    g.serialize(destination=str(path),format='turtle')
    assert len(Graph().parse(str(path),format='turtle'))>0

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    np.random.seed(args.seed)
    data,emb,src,reps=prepare_all(args)
    y=data['label'].to_numpy(dtype=int)
    ev=data['event_id'].astype(str).to_numpy()
    assert set(y)=={0,1}
    # Available features and no supervised target columns
    SHAPE=['t_source_length','t_source_wordcount','t_source_qmarks',
           't_source_hashtagcount','t_source_links','t_source_exclamation']
    configs={
      'C_Interaction_Semantic': SHAPE,
      'D_Interaction_Graph': SHAPE+old.GRAPH,
      'E_Interaction_RawEvidence': SHAPE+old.GRAPH+old.RAW,
      'F_Interaction_OntologyV11': SHAPE+old.GRAPH+old.RAW+old.RULES,
      'G_Interaction_OntologyV12': SHAPE+old.GRAPH+old.RAW+old.RULES+EV_RULES,
      'H_Interaction_OntologyV13': SHAPE+old.GRAPH+old.RAW+old.RULES+EV_RULES+O13_RULES,
      'I_StabilityPruned_Ontology': SHAPE+old.GRAPH+old.RAW+old.RULES+EV_RULES+O13_RULES,
      'K_Interaction_OntologyV14': SHAPE+old.GRAPH+old.RAW+old.RULES+RULES14,
      'L_Stable_OntologyV14': SHAPE+old.GRAPH+old.RAW+old.RULES+RULES14,
      'M_CatBoost_OntologyV14': SHAPE+old.GRAPH+old.RAW+old.RULES+RULES14,
    }
    assert not any('label' in c.lower() or 'rumor_ratio' in c.lower() for val in configs.values() for c in val)
    source_txt,context_txt=make_texts(data,reps)
    charvec=HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**16,
                              alternate_sign=False,norm='l2',dtype=np.float32)
    # HashingVectorizer is stateless (no training of vocabulary or IDF on test)
    hsource=charvec.transform(source_txt)
    hcontext=charvec.transform(context_txt)
    events=np.unique(ev)
    if args.events!='all':
        requested={x.strip() for x in args.events.split(',')}
        if not requested.issubset(set(events)):
            raise ValueError(f'Unknown event: {requested-set(events)}. Options {list(events)}')
        events=np.array(sorted(requested))
    if len(np.unique(ev))<3:raise ValueError('At least 3 distinct events are required')
    predictions=[]; metrics_rows=[]; audit_rows=[]; feature_rows=[]; branch_diagnostics=[]
    # Restart-safe fold checkpoint. The manifest prevents mixing incompatible runs.
    manifest_path=out/'experiment_signature.json'
    manifest_data={
        'signature_version':'v14.0', 'seed':int(args.seed),
        'max_replies':int(args.max_replies), 'no_minilm':bool(args.no_minilm),
        'fast':bool(args.fast), 'min_gain':float(args.min_gain),
        'dataset':hashlib.sha256(('\n'.join(data.thread_id.astype(str)+'|'+data.event_id.astype(str)+'|'+data.label.astype(str))).encode()).hexdigest(),
        'graph':hashlib.sha256(data[old.GRAPH].to_numpy(dtype=np.float32).tobytes()).hexdigest(),
        'ontology':hashlib.sha256(data[EV_RULES+O13_RULES].to_numpy(dtype=np.float32).tobytes()).hexdigest(),
        'v14_source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'v14_rules_source':hashlib.sha256(Path(__file__).with_name('pheme_v14_rules.py').read_bytes()).hexdigest(),
        'v11_extractor_source':hashlib.sha256(Path(__file__).with_name('pheme_v11_features.py').read_bytes()).hexdigest(),
        'rules14':hashlib.sha256(data[RULES14].to_numpy(dtype=np.float32).tobytes()).hexdigest(),
    }
    checkpoints={
        'metrics_per_event.csv':'metrics_rows',
        'inner_selection_audit.csv':'audit_rows',
        'test_predictions_by_thread.csv':'predictions',
        'feature_importance.csv':'feature_rows',
        'forced_branch_diagnostics.csv':'branch_diagnostics',
    }
    complete=set()
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest_data:
            raise ValueError('Cannot resume incompatible experiment. Choose a new OUTPUT_PREFIX or disable RESUME.')
        if all((out/f).exists() for f in checkpoints):
            cached={key:pd.read_csv(out/f).to_dict('records') for f,key in checkpoints.items()}
            metrics_rows=cached['metrics_rows'];audit_rows=cached['audit_rows']
            predictions=cached['predictions'];feature_rows=cached['feature_rows']
            branch_diagnostics=cached['branch_diagnostics']
            # Each finished fold must include ALL seven configurations and a selection audit.
            for e in events:
                nconfigs=len({r['Config'] for r in metrics_rows if r['event']==e})
                n_audit=sum(1 for r in audit_rows if r['event']==e)
                n_predictions=sum(1 for r in predictions if r['event']==e)
                if nconfigs==len(CONFIGS) and n_audit==len(CONFIGS) and n_predictions>=len(CONFIGS):
                    complete.add(e)
            print('Resuming complete outer test events:',sorted(complete),flush=True)
    else:
        # A fresh run never silently reuses old results with the same prefix.
        for f in checkpoints:
            (out/f).unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest_data,indent=2))
    def save_partial():
        for fn,varname in checkpoints.items():
            rows={'metrics_rows':metrics_rows,'audit_rows':audit_rows,
                  'predictions':predictions,'feature_rows':feature_rows,
                  'branch_diagnostics':branch_diagnostics}[varname]
            if rows:
                tmp=out/(fn+'.tmp')
                pd.DataFrame(rows).to_csv(tmp,index=False)
                os.replace(tmp,out/fn)
    for held in events:
        if held in complete:
            print('Skip completed event:',held,flush=True)
            continue
        print('\n=== OUTER HELD-OUT EVENT:',held,'===',flush=True)
        test=np.flatnonzero(ev==held);train=np.flatnonzero(ev!=held)
        inner_events=np.unique(ev[train]);
        if len(inner_events)<2:raise ValueError('At least two training events required')
        oof={name:np.full(len(data),np.nan,np.float32) for name in ['word_source','word_context','minilm',*configs]}
        # Strict nested event-level OOF: NO outer test labels or features for supervised fits
        for innerheld in inner_events:
            itrain=train[ev[train]!=innerheld];ival=train[ev[train]==innerheld]
            print(' inner event:',innerheld,'train=',len(itrain),'val=',len(ival),flush=True)
            oof['word_source'][ival]=clf_text(itrain,ival,source_txt,y,ev,hsource,args.seed)
            oof['word_context'][ival]=clf_text(itrain,ival,context_txt,y,ev,hcontext,args.seed)
            if src is not None:oof['minilm'][ival]=clf_emb(itrain,ival,src,y,ev,args.seed)
            else:oof['minilm'][ival]=oof['word_source'][ival]
            projection=train_only_projection(itrain,ival,emb,source_txt,context_txt,fast=args.fast)
            for name,cols in configs.items():
                pp,_=clf_dense(itrain,ival,data,cols,emb,y,ev,args.seed,low_memory=args.fast,projection=projection,
                    variant=('v14_stable' if name=='L_Stable_OntologyV14' else
                             'regularized' if name=='I_StabilityPruned_Ontology' else
                             'catboost' if name=='M_CatBoost_OntologyV14' else
                             'v14' if name=='K_Interaction_OntologyV14' else 'v12'))
                oof[name][ival]=pp
        # Base selection only from inner OOF
        family={
          'A_WordChar_Source':[(1.,0.,0.)],
          'B_Text_Reply_MiniLM':[(1.,0.,0.),(.7,.3,0.),(.4,.6,0.),(0.,1.,0.),
                                 (.7,0.,.3),(.4,.3,.3),(.2,.4,.4),(0.,0.,1.)],
        }
        base_selection={};p_oof=None
        for key,opts in family.items():
            best=None
            for ws,wc,wm in opts:
                p=ws*oof['word_source'][train]+wc*oof['word_context'][train]+wm*oof['minilm'][train]
                t,score=tune(y[train],p,ev[train])
                # prefer simplest on ties
                rank=(score,-abs(t-.5),ws)
                if best is None or rank>best['rank']:
                    best={'rank':rank,'weights':(ws,wc,wm),'threshold':t,'inner_score':score,'oof':p}
            base_selection[key]=best
        base=base_selection['B_Text_Reply_MiniLM'];p_oof=base['oof']
        # Only inner train event OOF selects branch weight, threshold, model family
        decisions={}
        base_thr,base_sc=tune(y[train],p_oof,ev[train])
        for name in configs:
            # Guard against spurious event-specific metadata uplift. A *positive* alpha
            # must improve the inner robust score by a precommitted minimum margin AND
            # must match/exceed the baseline on most training validation events.
            best={'rank':(base_sc,0.),'alpha':0.,'threshold':base_thr,'inner_score':base_sc}
            base_per_event={ee:metrics(y[train][ev[train]==ee],p_oof[ev[train]==ee],base_thr)['MacroF1']
                            for ee in inner_events}
            for alpha in (.15,.3,.5,.7,1.):
                p=(1-alpha)*p_oof+alpha*oof[name][train]
                t,score=tune(y[train],p,ev[train])
                agreement=sum(metrics(y[train][ev[train]==ee],p[ev[train]==ee],t)['MacroF1']>=base_per_event[ee]-0.002
                              for ee in inner_events)
                if (score>=base_sc+args.min_gain and agreement>=int(np.ceil(.6*len(inner_events)))
                    and (score,-alpha)>best['rank']):
                    best={'rank':(score,-alpha),'alpha':alpha,'threshold':t,'inner_score':score}
            decisions[name]=best
        # Outer model fitted on full outer TRAIN events; test used only for prediction
        final={}
        final['word_source']=clf_text(train,test,source_txt,y,ev,hsource,args.seed)
        final['word_context']=clf_text(train,test,context_txt,y,ev,hcontext,args.seed)
        final['minilm']=clf_emb(train,test,src,y,ev,args.seed) if src is not None else final['word_source']
        projection=train_only_projection(train,test,emb,source_txt,context_txt,fast=args.fast)
        for name,cols in configs.items():
            pp,model=clf_dense(train,test,data,cols,emb,y,ev,args.seed,low_memory=args.fast,projection=projection,
                variant=('v14_stable' if name=='L_Stable_OntologyV14' else
                             'regularized' if name=='I_StabilityPruned_Ontology' else
                             'catboost' if name=='M_CatBoost_OntologyV14' else
                             'v14' if name=='K_Interaction_OntologyV14' else 'v12'))
            final[name]=pp
            if name in ('E_Interaction_RawEvidence','F_Interaction_OntologyV11','G_Interaction_OntologyV12',
                         'H_Interaction_OntologyV13','I_StabilityPruned_Ontology',
                         'K_Interaction_OntologyV14','L_Stable_OntologyV14','M_CatBoost_OntologyV14'):
                # importance for ordinary original columns only; semantic PCA components first
                selected=model._v13_selected
                col_imp=np.array(model.feature_importances_)[-len(selected):] if selected else []
                for c,im in zip(selected,col_imp):
                    feature_rows.append({'event':held,'Config':name,'feature':c,'importance_split':int(im)})
        for key,best in base_selection.items():
            ws,wc,wm=best['weights']; pp=ws*final['word_source']+wc*final['word_context']+wm*final['minilm']
            m=metrics(y[test],pp,best['threshold'])
            metrics_rows.append(dict(event=held,Config=key,alpha=0.,threshold=best['threshold'],inner_score=best['inner_score'],**m))
            for j,p in zip(test,pp):predictions.append({'event':held,'Config':key,'thread_id':str(data.at[j,'thread_id']),'y_true':int(y[j]),'p_rumor':float(p),'threshold':best['threshold'],'alpha':0.})
            audit_rows.append(dict(event=held,Config=key,selected_alpha=0.,inner_score=best['inner_score'],threshold=best['threshold'],text_weights=json.dumps(best['weights'])))
        ws,wc,wm=base['weights'];pbase=ws*final['word_source']+wc*final['word_context']+wm*final['minilm']
        for name,sel in decisions.items():
            alpha=sel['alpha'];pp=(1-alpha)*pbase+alpha*final[name]
            m=metrics(y[test],pp,sel['threshold'])
            metrics_rows.append(dict(event=held,Config=name,alpha=alpha,threshold=sel['threshold'],inner_score=sel['inner_score'],**m))
            for j,p in zip(test,pp):predictions.append({'event':held,'Config':name,'thread_id':str(data.at[j,'thread_id']),'y_true':int(y[j]),'p_rumor':float(p),'threshold':sel['threshold'],'alpha':float(alpha)})
            audit_rows.append(dict(event=held,Config=name,selected_alpha=alpha,inner_score=sel['inner_score'],threshold=sel['threshold'],text_weights=json.dumps(base['weights'])))
            # IMPORTANT: pre-committed diagnostics do NOT select the reported winner.
            for diag_alpha in (.5,1.):
                pdiag=(1-diag_alpha)*pbase+diag_alpha*final[name]
                diag_inner=(1-diag_alpha)*p_oof+diag_alpha*oof[name][train]
                diag_thr,_=tune(y[train],diag_inner,ev[train])
                dmet=metrics(y[test],pdiag,diag_thr)
                branch_diagnostics.append(dict(event=held,Config=name,alpha_fixed=diag_alpha,
                    threshold_train_only=diag_thr,**dmet))
        # J is selected by inner-event OOF only, never by the outer test scores.
        # A small complexity penalty discourages feature-heavy overfitting.
        candidates={'B_Text_Reply_MiniLM':{'score':base['inner_score'],'threshold':base['threshold'],'alpha':0.}}
        candidates.update({name:{'score':val['inner_score'],'threshold':val['threshold'],
                            'alpha':val['alpha']} for name,val in decisions.items()
                          if name not in ('K_Interaction_OntologyV14','L_Stable_OntologyV14','M_CatBoost_OntologyV14')})
        penalties={'B_Text_Reply_MiniLM':0.,'C_Interaction_Semantic':.0002,
                   'D_Interaction_Graph':.0005,'E_Interaction_RawEvidence':.0009,
                   'F_Interaction_OntologyV11':.0012,'G_Interaction_OntologyV12':.0018,
                   'H_Interaction_OntologyV13':.0022,'I_StabilityPruned_Ontology':.0017}
        winner=max(candidates,key=lambda n:(candidates[n]['score']-penalties[n],-penalties[n]))
        picked=candidates[winner]
        pp=pbase if winner=='B_Text_Reply_MiniLM' else (1-picked['alpha'])*pbase+picked['alpha']*final[winner]
        md=metrics(y[test],pp,picked['threshold'])
        jname='J_EventCV_Selected'
        metrics_rows.append(dict(event=held,Config=jname,alpha=picked['alpha'],
                 threshold=picked['threshold'],inner_score=picked['score'],**md))
        for j,p in zip(test,pp):
            predictions.append({'event':held,'Config':jname,'thread_id':str(data.at[j,'thread_id']),
                  'y_true':int(y[j]),'p_rumor':float(p),'threshold':picked['threshold'],
                  'alpha':float(picked['alpha'])})
        audit_rows.append(dict(event=held,Config=jname,selected_alpha=picked['alpha'],
           inner_score=picked['score'],threshold=picked['threshold'],
           text_weights=json.dumps({'chosen_family':winner,'candidates':candidates,
                                    'complexity_penalties':penalties})))
        # Independent v14 family selector. PRECOMMITTED candidates only.
        # It compares v11 ontology against three v14 strategies; never uses held-out labels.
        v14_candidates={n:candidates[n] for n in (
            'B_Text_Reply_MiniLM','C_Interaction_Semantic',
            'D_Interaction_Graph','F_Interaction_OntologyV11')}
        v14_candidates.update({n:{'score':v['inner_score'],'threshold':v['threshold'],'alpha':v['alpha']}
            for n,v in decisions.items() if n in (
            'K_Interaction_OntologyV14','L_Stable_OntologyV14','M_CatBoost_OntologyV14')})
        penalty14={'B_Text_Reply_MiniLM':0.,'C_Interaction_Semantic':.0002,
           'D_Interaction_Graph':.0005,'F_Interaction_OntologyV11':.0012,
           'K_Interaction_OntologyV14':.0015,'L_Stable_OntologyV14':.0017,
           'M_CatBoost_OntologyV14':.0025}
        win14=max(v14_candidates,key=lambda nm:(v14_candidates[nm]['score']-penalty14[nm],-penalty14[nm]))
        chosen14=v14_candidates[win14]
        pred14=(pbase if win14=='B_Text_Reply_MiniLM' else
            (1-chosen14['alpha'])*pbase+chosen14['alpha']*final[win14])
        mm14=metrics(y[test],pred14,chosen14['threshold'])
        nm14='N_EventCV_Selected_v14'
        metrics_rows.append(dict(event=held,Config=nm14,alpha=chosen14['alpha'],
                 threshold=chosen14['threshold'],inner_score=chosen14['score'],**mm14))
        for j,z in zip(test,pred14):
            predictions.append(dict(event=held,Config=nm14,thread_id=str(data.at[j,'thread_id']),
               y_true=int(y[j]),p_rumor=float(z),threshold=chosen14['threshold'],
               alpha=float(chosen14['alpha'])))
        audit_rows.append(dict(event=held,Config=nm14,selected_alpha=chosen14['alpha'],
           inner_score=chosen14['score'],threshold=chosen14['threshold'],
           text_weights=json.dumps({'chosen_family':win14,'candidates':v14_candidates,
             'complexity_penalties':penalty14})))
        print(' PRECOMMITTED v14 family:',win14,'score:',round(chosen14['score'],4),flush=True)
        print(' PRECOMMITTED inner-event selected family:',winner,'score:',round(picked['score'],4),flush=True)
        save_partial()
        print(' Outer event saved:',held,flush=True)
    mdf=pd.DataFrame(metrics_rows); mdf.to_csv(out/'metrics_per_event.csv',index=False)
    pd.DataFrame(audit_rows).to_csv(out/'inner_selection_audit.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'test_predictions_by_thread.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(out/'feature_importance.csv',index=False)
    pd.DataFrame(branch_diagnostics).to_csv(out/'forced_branch_diagnostics.csv',index=False)
    summary=mdf.groupby('Config').agg(MacroF1_mean=('MacroF1','mean'),MacroF1_std=('MacroF1','std'),
               Recall_mean=('Recall','mean'),RumorF1_mean=('RumorF1','mean'),
               AUPRC_mean=('AUPRC','mean'),FN_total=('FN','sum'),FP_total=('FP','sum'),
               alpha_positive_events=('alpha',lambda v:int(np.sum(np.asarray(v)>0))),n_events=('event','nunique')).reset_index()
    summary.to_csv(out/'summary_by_config.csv',index=False)
    # Paired, within-event thread bootstrap is DESCRIPTIVE ONLY; not a test across new events.
    pred_df=pd.DataFrame(predictions)
    rng=np.random.default_rng(42)
    comparisons=[]
    for left,right in [
        ('G_Interaction_OntologyV12','E_Interaction_RawEvidence'),
        ('H_Interaction_OntologyV13','F_Interaction_OntologyV11'),
        ('I_StabilityPruned_Ontology','F_Interaction_OntologyV11'),
        ('J_EventCV_Selected','B_Text_Reply_MiniLM'),
        ('G_Interaction_OntologyV12','F_Interaction_OntologyV11'),
        ('F_Interaction_OntologyV11','E_Interaction_RawEvidence'),
        ('K_Interaction_OntologyV14','F_Interaction_OntologyV11'),
        ('L_Stable_OntologyV14','F_Interaction_OntologyV11'),
        ('M_CatBoost_OntologyV14','F_Interaction_OntologyV11'),
        ('N_EventCV_Selected_v14','J_EventCV_Selected')]:
        series=[]
        for e in events:
            xx=pred_df[(pred_df.event==e)&(pred_df.Config==left)].sort_values('thread_id')
            yy=pred_df[(pred_df.event==e)&(pred_df.Config==right)].sort_values('thread_id')
            assert np.array_equal(xx.thread_id.to_numpy(),yy.thread_id.to_numpy())
            assert np.array_equal(xx.y_true.to_numpy(),yy.y_true.to_numpy())
            yt=xx.y_true.to_numpy();pl=(xx.p_rumor.to_numpy()>=xx.threshold.to_numpy()).astype(int)
            pr=(yy.p_rumor.to_numpy()>=yy.threshold.to_numpy()).astype(int)
            series.append((yt,pl,pr))
        def dscore(pair,inds):
            yt,pl,pr=pair
            return f1_score(yt[inds],pl[inds],average='macro',zero_division=0)-f1_score(yt[inds],pr[inds],average='macro',zero_division=0)
        obs=float(np.mean([dscore(x,np.arange(len(x[0]))) for x in series]))
        boot=[]
        for _ in range(400):
            boot.append(np.mean([dscore(x,rng.integers(0,len(x[0]),len(x[0]))) for x in series]))
        comparisons.append({'model':left,'comparator':right,'delta_macro_f1':obs,
             'within_event_bootstrap_lower95':float(np.percentile(boot,2.5)),
             'within_event_bootstrap_upper95':float(np.percentile(boot,97.5)),
             'interpretation':'Conditional within-event CI; NOT uncertainty across new events'})
    pd.DataFrame(comparisons).to_csv(out/'paired_bootstrap_diagnostics.csv',index=False)
    ontology_rdf(pd.read_csv(out/'ontology_v12_evidence_provenance.csv',dtype={'thread_id':str}),out)
    ontology_rdf_v14(pd.read_csv(out/'ontology_v14_provenance.csv',dtype={'thread_id':str}),out)
    with open(out/'run_metadata.json','w') as f:
        json.dump({'seed':args.seed,'events_tested':list(events),'n_threads':int(len(data)),
                   'max_replies':args.max_replies,'uses_minilm':bool(src is not None),
                   'split_policy':'nested leave-one-event-out','ontology_uses_test_labels':False,'v13_train_only_feature_filter':True,'v14_label_blind_evidence':True,
                   'limit':'retrospective entire-thread classification; no causal/truth inference',
                   'checkpoint_resume':bool(args.resume),'model_choice':'inner events only',
                   'ontology_ablation':'F vs E; K/L/M vs F; N inner event selected; alpha>0 indicates branch use; source/reply evidence is not proof of truth'},f,indent=2)
    print('\n===== SUMMARY =====\n',summary.to_string(index=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pheme',required=True);parser.add_argument('--graph',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache',default='/kaggle/working/pheme_v12_embeddings')
    parser.add_argument('--events',default='all')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-replies',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--no-minilm',action='store_true')
    parser.add_argument('--fast',action='store_true')
    parser.add_argument('--min-gain',type=float,default=.003)
    parser.add_argument('--resume',action='store_true',help='Skip completed outer events only when dataset and settings match')
    args=parser.parse_args();run(args)

if __name__=='__main__': main()
