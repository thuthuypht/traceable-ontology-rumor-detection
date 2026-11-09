"""PHEME v11 - Source/Reply-aware, evidence-grounded Ontology LOEO.

Leakage discipline:
  - No test-event labels or pseudo-labels enter features or tuning.
  - All engineered features use only a thread's text, reply graph and timestamps.
  - Global MiniLM encoder is frozen and pretrained; TF-IDF and classifiers fit on training events.
  - 4 inner held-out events choose all mixing weights and thresholds.
  - This is retrospective full-thread rumor classification, not early detection.

Ontology rule activations are explicitly materialized Python rule consequences and RDF
provenance, NOT claims of OWL-DL, formal truth verification, or causality.
"""
from __future__ import annotations
import argparse, hashlib, json, re, os, zipfile, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, recall_score, precision_score, accuracy_score,
    average_precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

GRAPH = ['thread_depth','num_nodes','num_edges','avg_branching_factor','max_branching_factor',
 'source_reply_count','leaf_ratio','avg_depth','source_pagerank','avg_pagerank',
 'source_centrality','avg_centrality','unique_users','orphan_fraction',
 'cross_user_reply_ratio','reply_time_span_log','reply_time_median_log']

# Fixed a priori, no PHEME test-label-driven editing. These are linguistic cues,
# not automatically reliable stance classifications. Specific query patterns
# deliberately avoid using every '?' as a claim challenge.
CUES = {
 'denial': re.compile(r"\b(?:fake|hoax|false|untrue|fabricat(?:ed|ion)|debunk(?:ed|ing)?|misleading|not true|isn.t true|incorrect|not accurate|no evidence|no proof|made up|this is a lie|rumou?rs?)\b",re.I),
 'query': re.compile(r"\b(?:is (?:this|that|it) (?:true|real|confirmed)|any (?:proof|evidence|source)|where.s (?:the|your) (?:source|proof)|can (?:anyone|somebody) (?:confirm|verify)|has (?:this|it) been confirmed|how (?:do|can) (?:you|we) know|source\?|verified\?|proof\?)",re.I),
 'support': re.compile(r"\b(?:confirmed|verified|officially confirmed|it.s true|this is true|indeed true|accurate report|can confirm|yes,? (?:it is|that is) true)\b",re.I),
 'uncertainty': re.compile(r"\b(?:unconfirmed|unverified|reportedly|allegedly|apparently|might be|may be|not sure|unclear|rumou?rs?|seems like|could be)\b",re.I),
 'citation': re.compile(r"(?:https?://\S+|www\.\S+|\b(?:reuters|ap news|bbc|associated press|official statement|according to)\b)",re.I),
}
RAW = ['e_n_replies_log','e_n_authors_log','e_deny_rate','e_query_rate','e_support_rate',
 'e_uncertainty_rate','e_citation_rate','e_direct_deny_rate','e_direct_query_rate',
 'e_early_deny_rate','e_early_query_rate','e_denial_author_ratio',
 'e_query_author_ratio','e_challenge_author_count','e_cross_author_conflict','e_cited_denial_rate','e_cited_support_rate',
 'e_source_uncertainty','e_source_citation','e_direct_citation_rate',
 'e_challenge_ratio','e_dispute_diversity','e_reply_coverage','e_semantic_sim_mean',
 'e_semantic_sim_std','e_semantic_sim_min','e_semantic_novelty_rate',
 'e_semantic_direct_sim','e_semantic_early_late_shift','e_semantic_available']
RULES = ['r_direct_challenge','r_independent_challenge','r_cited_denial',
 'r_support_deny_conflict','r_early_challenge','r_uncertainty_under_query',
 'r_query_without_cited_support','r_semantic_challenge','r_multi_signal_dispute',
 'r_soft_direct_challenge','r_soft_independent_dispute','r_soft_cited_denial',
 'r_soft_ambiguity','r_soft_semantic_dispute','r_soft_evidence_strength']

RULE_DESCRIPTIONS = {
 'r_direct_challenge': 'Reply directly to the source contains denial or verification-request cue',
 'r_independent_challenge': 'At least two distinct reply authors use denial or verification-request cues',
 'r_cited_denial': 'At least one denial-cue reply also includes an external citation cue',
 'r_support_deny_conflict': 'Support and denial cues come from distinct reply authors',
 'r_early_challenge': 'One of the first three sampled replies uses denial or verification-request cue',
 'r_uncertainty_under_query': 'Source includes an uncertainty cue and reply asks for evidence',
 'r_query_without_cited_support': 'Verification query exists without a sampled citation-bearing support cue; scope is sampled replies only',
 'r_semantic_challenge': 'Denial cue together with low source-reply embedding cosine similarity (not logical contradiction)',
 'r_multi_signal_dispute': 'At least two independent types of dispute cue were activated',
 'r_soft_direct_challenge': 'Continuous direct challenge intensity from independently identified reply cues',
 'r_soft_independent_dispute': 'Continuous unique-author-weighted dispute intensity',
 'r_soft_cited_denial': 'Continuous citation-bearing denial evidence',
 'r_soft_ambiguity': 'Continuous interaction of source uncertainty and reply questioning',
 'r_soft_semantic_dispute': 'Continuous coupling of lexical challenge and semantic novelty',
 'r_soft_evidence_strength': 'Fixed rule-aggregated evidence strength; no target label is used',
}


def load_inputs(pheme_path,graph_path,max_replies=8):
    p=pd.read_csv(pheme_path,dtype={c:str for c in ['post_id','thread_id','reply_to','user_id','event_id']},low_memory=False)
    necessary={'post_id','thread_id','reply_to','user_id','event_id','text','time','label'}
    if not necessary.issubset(p): raise ValueError(f'Missing PHEME columns {necessary-set(p)}')
    p=p.drop_duplicates(['thread_id','post_id']).copy()
    p['post_id']=p.post_id.fillna('').str.strip();p['thread_id']=p.thread_id.fillna('').str.strip()
    p['reply_to']=p.reply_to.fillna('').str.strip()
    p['user_id']=p.user_id.fillna('').str.strip()
    # IDs already corrupted in floating-point scientific notation cannot be recovered.
    for c in ('thread_id','post_id','reply_to'):
        if p[c].str.contains(r'\d\.\d+e\+\d+',case=False,regex=True).any():
            raise ValueError(f'Potentially lossy scientific-notation Twitter IDs in {c}; use pheme_clean.csv')
    p['text']=p.text.fillna('').astype(str)
    p['label']=pd.to_numeric(p.label,errors='raise').astype(int)
    if not p.label.isin([0,1]).all():raise ValueError('Invalid target label')
    cons=p.groupby('thread_id').agg(nlab=('label','nunique'),nev=('event_id','nunique'))
    if ((cons.nlab!=1)|(cons.nev!=1)).any():raise ValueError('Thread label/event inconsistency')
    roots=p[p.post_id==p.thread_id].drop_duplicates('thread_id')
    if len(roots)!=len(cons):raise ValueError('Not exactly one source per thread')
    roots=roots[['thread_id','event_id','label','text','user_id']].rename(columns={'text':'source_text','user_id':'source_user'})
    graph=pd.read_csv(graph_path,dtype={'thread_id':str})
    if not graph.thread_id.is_unique:raise ValueError('Duplicate thread_id in graph data')
    if not set(GRAPH).issubset(graph):raise ValueError(f'Graph schema mismatch {set(GRAPH)-set(graph)}')
    forbidden=[c for c in graph if re.search(r'(label|rumor_ratio|user_prior|target|gold)',c,re.I)]
    if forbidden:raise ValueError(f'Target-dependent graph fields detected: {forbidden}')
    df=roots.merge(graph[['thread_id',*GRAPH]],on='thread_id',validate='one_to_one',how='left').sort_values('thread_id').reset_index(drop=True)
    if len(df)!=len(roots) or df[GRAPH].isna().any().any():raise ValueError('Unmatched/missing graph features')
    p['time_dt']=pd.to_datetime(p.time,format='%a %b %d %H:%M:%S %z %Y',errors='coerce',utc=True)
    if p.time_dt.isna().mean()>.15:raise ValueError('Too many unparsable Twitter timestamps')
    reps=p[p.post_id!=p.thread_id].sort_values(['thread_id','time_dt','post_id'],kind='stable')
    reps=reps.groupby('thread_id',sort=False).head(int(max_replies)).copy()
    reps['rank']=reps.groupby('thread_id').cumcount()
    reps=reps.merge(df[['thread_id','source_user']],on='thread_id',validate='many_to_one')
    # For sampled historical replies, parent relation is to source only when exact IDs match.
    reps['direct']=reps.reply_to==reps.thread_id
    if set(reps.thread_id)-set(df.thread_id):raise RuntimeError('Misaligned sampled replies')
    df['source_uncertainty']=df.source_text.map(lambda s:int(bool(CUES['uncertainty'].search(s))))
    df['source_citation']=df.source_text.map(lambda s:int(bool(CUES['citation'].search(s))))
    for cue,pat in CUES.items():reps[cue]=reps.text.map(lambda s:int(bool(pat.search(s))))
    print(f'Input: {len(p):,} tweets, {len(df):,} threads, {len(reps):,} sampled replies, {df.event_id.nunique()} events')
    return df,reps


def build_lexical(df,reps):
    """All reply evidence is computed without ever accessing label or event_id."""
    by={t:g for t,g in reps.groupby('thread_id',sort=False)}
    records=[]; traces={}
    for t,su,us,sc in df[['thread_id','source_user','source_uncertainty','source_citation']].itertuples(index=False,name=None):
        g=by.get(t)
        if g is None:
            records.append(dict(thread_id=t,**{n:0. for n in RAW if not n.startswith('e_semantic_')}))
            traces[t]=[];continue
        n=len(g);authors=set(g.user_id)-{''}
        da=set(g.loc[g.denial==1,'user_id'])-{''}
        qa=set(g.loc[g['query']==1,'user_id'])-{''}
        sa=set(g.loc[g.support==1,'user_id'])-{''}
        challenge=set(da)|set(qa)
        direct=g[g.direct];early=g.iloc[:3]
        rate=lambda x:float(x)/max(n,1)
        share=lambda x:float(x)/max(len(authors),1)
        rr={
            'e_n_replies_log':np.log1p(n),'e_n_authors_log':np.log1p(len(authors)),
            'e_deny_rate':rate(g.denial.sum()),'e_query_rate':rate(g['query'].sum()),
            'e_support_rate':rate(g.support.sum()),'e_uncertainty_rate':rate(g.uncertainty.sum()),
            'e_citation_rate':rate(g.citation.sum()),
            'e_direct_deny_rate':rate(direct.denial.sum()),'e_direct_query_rate':rate(direct['query'].sum()),
            'e_early_deny_rate':rate(early.denial.sum()),'e_early_query_rate':rate(early['query'].sum()),
            'e_denial_author_ratio':share(len(da)),'e_query_author_ratio':share(len(qa)),
            'e_challenge_author_count':float(len(challenge)),
            'e_cross_author_conflict':float(bool(da and sa and any(a!=b for a in da for b in sa))),
            'e_cited_denial_rate':rate(((g.denial==1)&(g.citation==1)).sum()),
            'e_cited_support_rate':rate(((g.support==1)&(g.citation==1)).sum()),
            'e_source_uncertainty':float(us),'e_source_citation':float(sc),
            'e_direct_citation_rate':rate(direct.citation.sum()),
            'e_challenge_ratio':rate((g.denial|g['query']).sum()),
            'e_dispute_diversity':share(len(challenge)),
            'e_reply_coverage':float(n)/max(int(df.loc[df.thread_id==t,'num_nodes'].iloc[0])-1,1),
        }
        records.append(dict(thread_id=t,**rr))
        traces[t]=[{'post_id':str(x.post_id),'user_id':str(x.user_id),'reply_to':str(x.reply_to),
              'direct':bool(x.direct),'denial':int(x.denial),'query':int(getattr(x,'query')),
              'support':int(x.support),'citation':int(x.citation),'uncertainty':int(x.uncertainty),
              'rank':int(x.rank)} for x in g.itertuples(index=False)]
    x=pd.DataFrame(records).set_index('thread_id').reindex(df.thread_id)
    if x.isna().any().any():raise RuntimeError('Missing lexical evidence features')
    x.index=df.index
    return x,traces


def get_embeddings(df,reps,out,batch_size=64,model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Frozen pretrained sentence encoder; caches in writable Kaggle working dir."""
    from sentence_transformers import SentenceTransformer
    # Deterministic alignment under fixed sorted source/reply rows.
    strings=df.source_text.astype(str).tolist()+reps.text.astype(str).tolist()
    digest=hashlib.sha256((model_name+'\n'+'\n'.join(strings)).encode()).hexdigest()[:18]
    target=Path(out)/f'frozen_minilm_{digest}.npz';target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():
        dat=np.load(target)
        a,b=dat['sources'],dat['replies']
        if len(a)==len(df) and len(b)==len(reps) and np.isfinite(a).all():return a,b
    encoder=SentenceTransformer(model_name)
    embs=encoder.encode(strings,batch_size=batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float32)
    s,r=embs[:len(df)],embs[len(df):]
    np.savez_compressed(target,sources=s,replies=r)
    return s,r


def semantic_information(df,reps,source_emb,reply_emb):
    n=len(df);dim=int(source_emb.shape[1]);thread_to_i={t:i for i,t in enumerate(df.thread_id)}
    mean=np.zeros((n,dim),np.float32);first=np.zeros((n,dim),np.float32)
    sim=np.zeros((n,8),np.float32)
    has=np.zeros(n,bool)
    # rows of reps align with embeddings; group indices are positional not dataframe index.
    groups=defaultdict(list)
    for i,t in enumerate(reps.thread_id):groups[t].append(i)
    for t,indices in groups.items():
        j=thread_to_i[t];v=reply_emb[indices];ifempty=len(v)==0
        if ifempty:continue
        has[j]=True;mean[j]=v.mean(axis=0);first[j]=v[:min(3,len(v))].mean(axis=0)
        cs=np.clip(v@source_emb[j],-1,1)
        direct=reps.iloc[indices].direct.to_numpy(dtype=bool)
        early=cs[:min(3,len(cs))];late=cs[3:]
        sim[j]=[np.mean(cs),np.std(cs),np.min(cs),np.mean(cs<.40),
             np.mean(cs[direct]) if direct.any() else 0.,
             float(np.mean(late)-np.mean(early)) if len(late) else 0.,
             np.mean(cs>=.65),np.max(cs)]
    X=np.hstack([source_emb,mean,np.abs(source_emb-mean),sim,has[:,None].astype('float32')]).astype('float32')
    s_names=['e_semantic_sim_mean','e_semantic_sim_std','e_semantic_sim_min',
             'e_semantic_novelty_rate','e_semantic_direct_sim','e_semantic_early_late_shift']
    evidence=pd.DataFrame(sim[:,:6],columns=s_names,index=df.index)
    evidence['e_semantic_available']=has.astype(float)
    return X,evidence


def build_rules(x):
    """Materialize deterministic, label-blind rule activations with explicit provenance criteria.
    Rules are descriptive hypothesis features, NOT truth verification or DL entailment.
    """
    a=x;deny=a.e_deny_rate.to_numpy();query=a.e_query_rate.to_numpy()
    sup=a.e_support_rate.to_numpy();d_deny=a.e_direct_deny_rate.to_numpy()
    d_query=a.e_direct_query_rate.to_numpy();n=a.e_n_replies_log.to_numpy()
    author=a.e_dispute_diversity.to_numpy();early=a.e_early_deny_rate.to_numpy()+a.e_early_query_rate.to_numpy()
    citation=a.e_cited_denial_rate.to_numpy();uncert=a.e_source_uncertainty.to_numpy()
    novelty=a.e_semantic_novelty_rate.to_numpy();available=a.e_semantic_available.to_numpy()
    chall=deny+query
    result={
      'r_direct_challenge':(d_deny+d_query>0).astype(float),
      'r_independent_challenge':(a.e_challenge_author_count.to_numpy()>=2).astype(float),
      'r_cited_denial':(citation>0).astype(float),
      'r_support_deny_conflict':a.e_cross_author_conflict.to_numpy().astype(float),
      'r_early_challenge':(early>0).astype(float),
      'r_uncertainty_under_query':((uncert>0)&(query>0)).astype(float),
      'r_query_without_cited_support':((query>0)&(a.e_cited_support_rate.to_numpy()==0)).astype(float),
      'r_semantic_challenge':((deny>0)&(novelty>0)&(available>0)).astype(float),
      'r_multi_signal_dispute':(((d_deny+d_query>0).astype(int)+(citation>0).astype(int)+(early>0).astype(int)+(a.e_cross_author_conflict.to_numpy()>0).astype(int))>=2).astype(float),
      'r_soft_direct_challenge':np.clip(d_deny+d_query,0,1),
      'r_soft_independent_dispute':np.clip(chall*author,0,1),
      'r_soft_cited_denial':np.clip(citation*(1+author),0,1),
      'r_soft_ambiguity':np.clip(uncert*query,0,1),
      'r_soft_semantic_dispute':np.clip(chall*novelty*available,0,1),
      'r_soft_evidence_strength':np.clip(.30*(d_deny+d_query)+.25*chall*author+.25*citation+.20*query*uncert,0,1),
    }
    rr=pd.DataFrame(result,index=x.index)
    assert list(rr.columns)==RULES
    return rr


def prepare(pheme,graph,out,max_replies=8,encode=True,batch_size=64,cache_dir=None):
    df,reps=load_inputs(pheme,graph,max_replies)
    lexical,traces=build_lexical(df,reps)
    emb=None
    if encode:
        s,r=get_embeddings(df,reps,Path(cache_dir) if cache_dir else Path(out)/'embeddings',batch_size=batch_size)
        emb,sem=semantic_information(df,reps,s,r)
        lexical=pd.concat([lexical,sem],axis=1)
        source=s
    else:
        source=None
        for n in RAW:
            if n.startswith('e_semantic_'):lexical[n]=0.
    lexical=lexical[RAW].astype('float32')
    rules=build_rules(lexical).astype('float32')
    df=pd.concat([df,lexical,rules],axis=1)
    for group in (GRAPH,RAW,RULES):
        if not np.isfinite(df[group].to_numpy(dtype=float)).all():raise ValueError('Nonfinite model features')
    assert len(df)==df.thread_id.nunique() and not set(['label','event_id']).intersection(RAW+RULES)
    return df,emb,source,traces,reps


def weighted_train(y,events,idx):
    yy=y[idx];ev=events[idx];weights=np.zeros(len(idx),dtype=float)
    for e in np.unique(ev):
        for c in (0,1):
            m=(ev==e)&(yy==c)
            if m.any(): weights[m]=1/(2*len(np.unique(ev))*m.sum())
    if weights.mean()==0:raise ValueError('No weighted training examples')
    return weights/weights.mean()


def fit_text(train,test,txt,source_emb,y,ev,hash_char,seed):
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=20000,min_df=2,max_df=.98,
       sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
    a=vec.fit_transform([txt[i] for i in train]);b=vec.transform([txt[i] for i in test])
    a=hstack([a,hash_char[train]],format='csr');b=hstack([b,hash_char[test]],format='csr')
    w=weighted_train(y,ev,train)
    lr=LogisticRegression(solver='liblinear',C=1.,max_iter=400,random_state=seed)
    lr.fit(a,y[train],sample_weight=w)
    tf=lr.predict_proba(b)[:,1]
    if source_emb is None:return tf,tf
    scaler=StandardScaler().fit(source_emb[train]);a=np.clip(scaler.transform(source_emb[train]),-6,6);b=np.clip(scaler.transform(source_emb[test]),-6,6)
    lm=LogisticRegression(solver='liblinear',C=.5,max_iter=400,random_state=seed)
    lm.fit(a,y[train],sample_weight=w)
    return tf,lm.predict_proba(b)[:,1]


def fit_dense(train,test,mat,y,ev,seed,method='lightgbm'):
    w=weighted_train(y,ev,train)
    if method=='lightgbm':
        try:
            from lightgbm import LGBMClassifier
            m=LGBMClassifier(n_estimators=170,learning_rate=.035,num_leaves=11,max_depth=4,
             min_child_samples=60,reg_lambda=10.,colsample_bytree=.85,
             subsample=.85,subsample_freq=1,verbosity=-1,n_jobs=2,random_state=seed)
            m.fit(mat[train],y[train],sample_weight=w)
            return m.predict_proba(mat[test])[:,1]
        except ImportError: print('LightGBM unavailable: using regularized logistic regression')
    sc=StandardScaler().fit(mat[train]);a=np.clip(sc.transform(mat[train]),-7,7);b=np.clip(sc.transform(mat[test]),-7,7)
    m=LogisticRegression(solver='liblinear',C=.3,max_iter=500,random_state=seed)
    m.fit(a,y[train],sample_weight=w)
    return m.predict_proba(b)[:,1]


def measure(y,p,threshold):
    pred=(np.asarray(p)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(MacroF1=float(f1_score(y,pred,average='macro',zero_division=0)),
     RumorF1=float(f1_score(y,pred,zero_division=0)),Recall=float(recall_score(y,pred,zero_division=0)),
     Precision=float(precision_score(y,pred,zero_division=0)),Accuracy=float(accuracy_score(y,pred)),
     MCC=float(matthews_corrcoef(y,pred)),
     AUPRC=float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,
     ROC_AUC=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
     TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))


def score_event(y,p,ev,thr):
    """Robust event-balanced objective, independent of outer test data."""
    return float(_threshold_scores(y,p,ev,np.array([thr],float))[0])


def _threshold_scores(y,p,ev,thresholds):
    """Vectorized, exact Macro-F1 over many thresholds without sklearn call overhead."""
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=np.float64)
    ev=np.asarray(ev);thresholds=np.asarray(thresholds)
    scores=[]
    for e in np.unique(ev):
        select=ev==e
        y1=y[select,None].astype(bool)
        yp=p[select,None]>=thresholds[None,:]
        tp=np.count_nonzero(y1 & yp,axis=0).astype(float)
        tn=np.count_nonzero((~y1)&(~yp),axis=0).astype(float)
        fp=np.count_nonzero((~y1)&yp,axis=0).astype(float)
        fn=np.count_nonzero(y1&(~yp),axis=0).astype(float)
        pden=2*tp+fp+fn;nden=2*tn+fp+fn
        f1p=np.divide(2*tp,pden,out=np.zeros_like(tp),where=pden>0)
        f1n=np.divide(2*tn,nden,out=np.zeros_like(tn),where=nden>0)
        scores.append(.5*(f1p+f1n))
    a=np.vstack(scores)
    return .8*a.mean(axis=0)+.2*a.min(axis=0)


def choose_threshold(y,p,ev):
    thresholds=np.linspace(.15,.85,71)
    scores=_threshold_scores(y,p,ev,thresholds)
    i=max(range(len(thresholds)),key=lambda j:(scores[j],-abs(thresholds[j]-.5)))
    return float(thresholds[i]),float(scores[i])


def pick_source(oof,y,ev,ids,with_minilm):
    weights=(0.,.25,.5,.75,1.) if with_minilm else (0.,)
    best=None
    for w in weights:
        p=w*oof['minilm'][ids]+(1-w)*oof['tfidf'][ids]
        thr,score=choose_threshold(y[ids],p,ev[ids]);rank=(score,-abs(w-.25))
        if best is None or rank>best[0]:best=(rank,w,thr,score)
    return dict(weight=best[1],threshold=best[2],score=best[3])

def gate(pbase,pextra,y,ev,min_gain=.003,alphas=(0.,.1,.25,.4,.6)):
    bt,bs=choose_threshold(y,pbase,ev)
    opts=[]
    for alpha in alphas:
        mix=(1-alpha)*pbase+alpha*pextra;thr,sc=choose_threshold(y,mix,ev)
        opts.append((sc,-alpha,thr,alpha))
    best=max(opts)
    if best[3]==0 or best[0]<bs+min_gain:return dict(alpha=0.,threshold=bt,score=bs,delta=0.,unconstrained_best=float(best[0]-bs))
    return dict(alpha=float(best[3]),threshold=float(best[2]),score=float(best[0]),delta=float(best[0]-bs),unconstrained_best=float(best[0]-bs))

CONFIGS=['A_Source','B_Source_ReplyContext','C_Context_Graph','D_Context_Graph_RawEvidence','E_Context_Graph_RawEvidence_OntologyRules']

def evaluate(df,context,source_emb,out,events='all',seed=42,method='lightgbm',min_gain=.003):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    y=df.label.to_numpy(dtype=int);ev=df.event_id.to_numpy(dtype=str);tid=df.thread_id.to_numpy(dtype=str)
    txt=df.source_text.fillna('').astype(str).tolist()
    char=HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**14,norm='l2',alternate_sign=False,dtype=np.float32).transform(txt).tocsr()
    blocks={
      'C_Context_Graph':GRAPH,
      'D_Context_Graph_RawEvidence':GRAPH+RAW,
      'E_Context_Graph_RawEvidence_OntologyRules':GRAPH+RAW+RULES
    }
    mat={name:df[cols].to_numpy(dtype=np.float32) for name,cols in blocks.items()}
    requested=sorted(np.unique(ev)) if events=='all' else [v.strip() for v in events.split(',')]
    if not set(requested).issubset(set(ev)):raise ValueError('Unrecognized event names: '+str(set(requested)-set(ev)))
    all_rows=[];audit=[];predictions=[];forced_rows=[];inner_model=[]
    for held in requested:
        tr=np.flatnonzero(ev!=held);te=np.flatnonzero(ev==held)
        inner_events=np.unique(ev[tr]);assert len(inner_events)>=3
        pred_oof={key:np.full(len(df),np.nan,dtype=float) for key in ['tfidf','minilm','context',*blocks]}
        print(f'Outer test={held}  train={len(tr)}  test={len(te)}',flush=True)
        for val in inner_events:
            itr=np.flatnonzero((ev!=held)&(ev!=val));iva=np.flatnonzero(ev==val)
            assert not set(tid[itr])&set(tid[iva])
            tf,lm=fit_text(itr,iva,txt,source_emb,y,ev,char,seed)
            pred_oof['tfidf'][iva]=tf;pred_oof['minilm'][iva]=lm
            if context is not None:pred_oof['context'][iva]=fit_dense(itr,iva,context,y,ev,seed,method='logistic')
            else:pred_oof['context'][iva]=tf
            for name,array in mat.items():pred_oof[name][iva]=fit_dense(itr,iva,array,y,ev,seed,method)
            print('  Inner held-out:',val,flush=True)
        assert all(np.isfinite(p[tr]).all() for p in pred_oof.values())
        src=pick_source(pred_oof,y,ev,tr,source_emb is not None)
        p_src=(src['weight']*pred_oof['minilm'][tr]+(1-src['weight'])*pred_oof['tfidf'][tr])
        context_gate=gate(p_src,pred_oof['context'][tr],y[tr],ev[tr],min_gain)
        p_ctx=(1-context_gate['alpha'])*p_src+context_gate['alpha']*pred_oof['context'][tr]
        nested={name:gate(p_ctx,pred_oof[name][tr],y[tr],ev[tr],min_gain) for name in blocks}
        chosen={'A_Source':src['score'],'B_Source_ReplyContext':context_gate['score']}
        chosen.update({n:v['score'] for n,v in nested.items()})
        champion=max(CONFIGS,key=lambda n:(chosen[n],-CONFIGS.index(n)))
        for n,g in [('B_Source_ReplyContext',context_gate),*list(nested.items())]:
            inner_model.append(dict(event=held,Config=n,alpha=g['alpha'],score=g['score'],
                          best_delta=g['unconstrained_best'],selected_delta=g['delta']))
        tf,lm=fit_text(tr,te,txt,source_emb,y,ev,char,seed)
        baseline=src['weight']*lm+(1-src['weight'])*tf
        pcontext=fit_dense(tr,te,context,y,ev,seed,'logistic') if context is not None else tf
        p_B=(1-context_gate['alpha'])*baseline+context_gate['alpha']*pcontext
        outputs={'A_Source':(baseline,src['threshold'],0.),
                 'B_Source_ReplyContext':(p_B,context_gate['threshold'],context_gate['alpha'])}
        for name,array in mat.items():
            m=fit_dense(tr,te,array,y,ev,seed,method)
            g=nested[name];alpha=g['alpha']
            outputs[name]=((1-alpha)*p_B+alpha*m,g['threshold'],alpha)
            # Diagnostic forced-mix, fixed alpha=0.25 and a training-event threshold,
            # NEVER used to choose a champion after seeing test labels.
            fthr,_=choose_threshold(y[tr],.75*p_ctx+.25*pred_oof[name][tr],ev[tr])
            fm=measure(y[te],.75*p_B+.25*m,fthr)
            forced_rows.append(dict(event=held,Config=name,alpha_fixed=.25,threshold=fthr,**fm))
        for name,(prob,thr,alpha) in outputs.items():
            m=measure(y[te],prob,thr)
            all_rows.append(dict(event=held,seed=seed,Config=name,n_test=len(te),n_rumor_test=int(y[te].sum()),
                  threshold=thr,branch_alpha=alpha,champion=int(champion==name),inner_score=chosen[name],**m))
            for j,pr in zip(te,prob):predictions.append(dict(thread_id=tid[j],event=held,seed=seed,Config=name,
                y_true=int(y[j]),p_rumor=float(pr),y_pred=int(pr>=thr),alpha=alpha,threshold=thr))
        audit.append(dict(event=held,champion=champion,source_minilm_weight=src['weight'],
               source_threshold=src['threshold'],source_inner_score=src['score'],
               context_alpha=context_gate['alpha'],**{name+'_alpha':g['alpha'] for name,g in nested.items()}))
        pd.DataFrame(all_rows).to_csv(out/'metrics_per_event.csv',index=False)
        pd.DataFrame(audit).to_csv(out/'inner_selection_audit.csv',index=False)
        pd.DataFrame(predictions).to_csv(out/'test_predictions_by_thread.csv',index=False)
        pd.DataFrame(forced_rows).to_csv(out/'forced_branch_diagnostics.csv',index=False)
        pd.DataFrame(inner_model).to_csv(out/'inner_branch_diagnostics.csv',index=False)
        print(pd.DataFrame(all_rows).query('event == @held')[['Config','MacroF1','Recall','FN','FP','branch_alpha']].round(4).to_string(index=False),flush=True)
    met=pd.DataFrame(all_rows)
    summ=met.groupby('Config',sort=False).agg(MacroF1_mean=('MacroF1','mean'),
        MacroF1_std=('MacroF1','std'),Recall_mean=('Recall','mean'),
        AUPRC_mean=('AUPRC','mean'),FN_total=('FN','sum'),FP_total=('FP','sum'),
        branch_selected_folds=('branch_alpha',lambda s:int((s>0).sum())),
        champion_folds=('champion','sum'),n_events=('event','nunique')).reset_index()
    summ.to_csv(out/'summary_by_config.csv',index=False)
    print('SUMMARY\n',summ.round(4).to_string(index=False),flush=True)
    return summ


def serialize_ontology(df,traces,out,n_examples=6):
    try:
        from rdflib import Graph as RDFGraph,Namespace,Literal,URIRef
        from rdflib.namespace import RDF,RDFS,OWL,XSD,PROV
    except ImportError:
        print('rdflib missing, skipping RDF serialization');return
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    ex=Namespace('https://example.org/pheme-v11/')
    g=RDFGraph();g.bind('ex',ex);g.bind('prov',PROV);g.bind('owl',OWL)
    for name in ('Claim','Reply','EvidenceCue','Rule','RuleActivation'):
        g.add((ex[name],RDF.type,OWL.Class))
    for name in ('respondsTo','usesCue','activatesRule','derivedFromReply','aboutClaim'):
        g.add((ex[name],RDF.type,OWL.ObjectProperty))
    for name in ('value','ruleDescription','rank'):
        g.add((ex[name],RDF.type,OWL.DatatypeProperty))
    for name,desc in RULE_DESCRIPTIONS.items():
        g.add((ex[name],RDF.type,ex.Rule));g.add((ex[name],RDFS.comment,Literal(desc)))
    trace_out=[]
    positive=df[df[RULES[:9]].sum(axis=1)>0].head(n_examples)
    for _,row in positive.iterrows():
        t=str(row.thread_id);claim=ex['claim/'+t]
        g.add((claim,RDF.type,ex.Claim))
        for rep in traces.get(t,[]):
            node=ex['reply/'+t+'/'+rep['post_id']];g.add((node,RDF.type,ex.Reply))
            parent=ex['claim/'+t] if rep['reply_to']==t else ex['reply/'+t+'/'+rep['reply_to']]
            g.add((node,ex.respondsTo,parent))
            g.add((node,ex.rank,Literal(rep['rank'],datatype=XSD.integer)))
            for cue in ('denial','query','support','citation','uncertainty'):
                if rep[cue]:g.add((node,ex.usesCue,ex['cue/'+cue]))
        active=[]
        for rule in RULES:
            val=float(row[rule])
            if val<=0:continue
            activation=ex['activation/'+t+'/'+rule]
            g.add((activation,RDF.type,ex.RuleActivation))
            g.add((activation,ex.aboutClaim,claim))
            g.add((activation,ex.activatesRule,ex[rule]))
            g.add((activation,ex.value,Literal(val,datatype=XSD.double)))
            for rep in traces.get(t,[]):
                challenge=bool(rep['denial'] or rep['query'])
                if rule in ('r_direct_challenge','r_soft_direct_challenge'):
                    relevant=rep['direct'] and challenge
                elif rule in ('r_cited_denial','r_soft_cited_denial'):
                    relevant=bool(rep['denial'] and rep['citation'])
                elif rule=='r_support_deny_conflict':
                    relevant=bool(rep['denial'] or rep['support'])
                elif rule=='r_early_challenge':
                    relevant=rep['rank']<3 and challenge
                elif rule in ('r_uncertainty_under_query','r_query_without_cited_support','r_soft_ambiguity'):
                    relevant=bool(rep['query'])
                elif rule in ('r_semantic_challenge','r_soft_semantic_dispute'):
                    relevant=bool(rep['denial'])
                else:
                    relevant=challenge
                if relevant:
                    g.add((activation,ex.derivedFromReply,ex['reply/'+t+'/'+rep['post_id']]))
            active.append({'rule':rule,'value':val})
        trace_out.append({'thread_id':t,'active_rules':active,'reply_evidence':traces.get(t,[])})
    g.serialize(destination=str(out/'ontology_evidence_sample.ttl'),format='turtle')
    # Declarative SHACL constraints for independent structural validation.  They
    # validate the shape of materialized claims, not logical truth of heuristics.
    shapes='''@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <https://example.org/pheme-v11/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
ex:ActivationShape a sh:NodeShape ; sh:targetClass ex:RuleActivation ;
  sh:property [ sh:path ex:aboutClaim ; sh:minCount 1 ; sh:maxCount 1 ; sh:class ex:Claim ] ;
  sh:property [ sh:path ex:activatesRule ; sh:minCount 1 ; sh:maxCount 1 ; sh:class ex:Rule ] ;
  sh:property [ sh:path ex:value ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:double ] .
ex:ReplyShape a sh:NodeShape ; sh:targetClass ex:Reply ;
  sh:property [ sh:path ex:respondsTo ; sh:minCount 1 ; sh:maxCount 1 ] .
'''
    (out/'ontology_shapes_v11.ttl').write_text(shapes)
    # This separate TBox can be reused without instantiating specific PHEME posts.
    tb=RDFGraph()
    for triple in g:
        if triple[1] in (RDF.type,RDFS.comment) and (str(triple[0]).startswith(str(ex)) and
             ('/' not in str(triple[0])[len(str(ex)):] )):
            tb.add(triple)
    tb.serialize(destination=str(out/'ontology_schema_v11.ttl'),format='turtle')
    with (out/'ontology_explanations_sample.jsonl').open('w') as f:
        for d in trace_out:f.write(json.dumps(d)+'\n')
    (out/'ontology_rules_spec.json').write_text(json.dumps(RULE_DESCRIPTIONS,indent=2))
    print(f'RDF sample: {len(g)} triples, {len(trace_out)} claims')


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--pheme',required=True);pa.add_argument('--graph',required=True)
    pa.add_argument('--out',required=True);pa.add_argument('--events',default='all')
    pa.add_argument('--seed',type=int,default=42);pa.add_argument('--max-replies',type=int,default=8)
    pa.add_argument('--batch-size',type=int,default=64);pa.add_argument('--skip-minilm',action='store_true')
    pa.add_argument('--meta-method',choices=['lightgbm','logistic'],default='lightgbm')
    pa.add_argument('--min-gain',type=float,default=.003)
    pa.add_argument('--audit-only',action='store_true')
    pa.add_argument('--cache-dir',default=None)
    args=pa.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df,ctx,src,traces,reps=prepare(args.pheme,args.graph,out,args.max_replies,
                                   encode=not args.skip_minilm,batch_size=args.batch_size,cache_dir=args.cache_dir)
    feature_cols=GRAPH+RAW+RULES
    cov=pd.DataFrame([{'feature':c,'nonzero_fraction':float((df[c]!=0).mean()),
            'n_unique':int(df[c].nunique())} for c in RAW+RULES])
    cov.to_csv(out/'ontology_feature_coverage.csv',index=False)
    df[['thread_id',*RAW,*RULES]].to_csv(out/'ontology_v11_features_label_blind.csv',index=False)
    serialize_ontology(df,traces,out)
    # Deterministic label-permutation invariance of every evidence/graph input.
    rng=np.random.default_rng(123);original=df[feature_cols].to_numpy(copy=True)
    perm=df.copy();perm['label']=rng.permutation(perm.label.to_numpy())
    assert np.array_equal(original,perm[feature_cols].to_numpy())
    assert not any(re.search(r'(label|target|rumor_ratio|user_prior)',n,re.I) for n in feature_cols)
    print('PASS: label-blind feature invariance and target-derived column audit')
    if not args.audit_only:
        evaluate(df,ctx,src,out,args.events,args.seed,args.meta_method,args.min_gain)
    metadata={'version':'v11','seed':args.seed,'outer_protocol':'Leave-One-Event-Out (five PHEME events)',
        'inner_protocol':'Four training-event held-out folds for threshold and fusion weight selection',
        'features':'PHEME text/retrospective replies + per-thread label-blind graph and evidence rules',
        'stance_v9_used':False,
        'model':'Frozen pretrained MiniLM + training-only TF-IDF; logistic context and LightGBM metadata',
        'max_replies':args.max_replies,'ontology_semantics':'Materialized rule-based heuristic evidence, NOT OWL-DL entailment',
        'early_detection':False,'min_gain':args.min_gain,
        'major_caveat':'PHEME test events have been used repeatedly to develop prior versions; development results, not locked external validation'}
    (out/'run_metadata_v11.json').write_text(json.dumps(metadata,indent=2))
    (out/'README_RESULTS.md').write_text('# PHEME v11 Output\n\n'
        'A = source text; B = source plus reply-context embedding; C = B plus graph; '
        'D = C plus raw lexical/semantic evidence; E = D plus materialized Ontology rules.\n\n'
        'The **guarded** model sets branch alpha=0 when validation rejects evidence. '
        'This is not an Ontology improvement. `forced_branch_diagnostics.csv` deliberately '
        'uses fixed alpha=0.25 *only for diagnostic reporting*, never to pick the champion.\n\n'
        'Rules are transparent materialized evidence predicates, not truth-verifying OWL-DL inference. '
        'PHEME cross-event development results alone cannot establish publishable generalization.\n')

if __name__=='__main__':main()
