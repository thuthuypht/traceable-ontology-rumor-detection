"""Twitter15/16 one-run controlled cross-corpus replication (development follow-up).

Ontology here is a *fuzzy, explicit, structural evidence-rule feature* layer,
NOT an OWL-DL proof or a fact-verification system. No reply text is assumed.
Both transfer directions use source-domain-only training, inner OOF selection,
and target evaluation once per fitted, frozen configuration.
"""
from __future__ import annotations
import io, os, re, json, hashlib, zipfile, math, time, warnings
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (f1_score, recall_score, precision_score, accuracy_score,
                             average_precision_score, matthews_corrcoef, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics.pairwise import cosine_similarity

SEED=42
NUMERIC_GRAPH=['graph_nodes','graph_edges','graph_unique_users','graph_max_depth',
 'graph_mean_depth','graph_root_branches','graph_max_outdegree','graph_mean_outdegree',
 'graph_early_60_count','graph_early_60_unique_users','graph_early_60_frac',
 'graph_late_1440_count','graph_duration_min','graph_time_median','graph_time_iqr',
 'graph_leaf_frac','graph_coverage','graph_malformed_rows']
CLASSIC_RULES=['Rule_ClaimUncertainty','Rule_RapidBroadPropagation','Rule_MultiBranchCascade',
 'Rule_IndependentAmplification','Rule_PersistentDiffusion','Rule_UncertainRapidClaim',
 'Rule_ComplexAmplification','Rule_VerificationPriority']
LEX_KEYS=['lex_uncertainty','lex_question','lex_attribution','lex_hedging','lex_url','lex_mention','lex_hashtag','lex_caps','lex_length']
RULE_KEYS=['Ont_RapidDiffusion','Ont_SustainedDiffusion','Ont_MultiPathSpread','Ont_DecentralizedSpread',
 'Ont_DeepCascade','Ont_IndependentAmplification','Ont_ClaimVerificationCue','Ont_SourceAttribution',
 'Ont_UncertainEarlySpread','Ont_SustainedCascade','Ont_MultiSignalReview','Ont_ReliabilityWeight']

@dataclass
class Config:
    seed:int=42
    placebo_repeats:int=49
    use_minilm:bool=True
    minilm_model:str='sentence-transformers/all-MiniLM-L6-v2'
    minilm_batch_size:int=128
    cv_folds:int=3
    cat_iterations:int=360
    cat_depth:int=5
    cat_learning_rate:float=0.045
    near_duplicate_threshold:float=0.975
    output_prefix:str='Twitter1516_v17_OneRun_Results'
    workdir:str='/kaggle/working'
    input_root:str='/kaggle/input'
    allow_fallback_encoder:bool=False
    min_examples:int=100


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(2**20)
            if not b:break
            h.update(b)
    return h.hexdigest()


def normalize(s):
    s=str(s).lower()
    s=re.sub(r'https?://\S+',' URL ',s)
    s=re.sub(r'\s+',' ',s).strip()
    return s


def read_input_materialized(input_root, workdir, embedded_gzip_b64=None):
    """No labels/predictions from v16/PHEME or old target tests are loaded."""
    roots=[Path(input_root),Path(workdir)]
    paths=[]
    for root in roots:
        if root.exists():
            paths.extend(root.rglob('materialized_features.csv'))
    for p in sorted(set(paths),key=lambda x:len(str(x))):
        print('Loading:',p)
        return pd.read_csv(p,dtype={'source_id':str}),str(p),sha256_file(p)
    for root in roots:
        if root.exists():
            for p in sorted(root.rglob('*.zip')):
                try:
                    with zipfile.ZipFile(p) as z:
                        matches=[n for n in z.namelist() if n.endswith('materialized_features.csv') and not n.startswith('__MACOSX')]
                        if matches:
                            b=z.read(matches[0]);print('Loading CSV from:',p,'/',matches[0]);
                            return pd.read_csv(io.BytesIO(b),dtype={'source_id':str}),str(p)+':'+matches[0],hashlib.sha256(b).hexdigest()
                except (zipfile.BadZipFile, OSError):pass
    if embedded_gzip_b64:
        import gzip,base64
        b=gzip.decompress(base64.b64decode(embedded_gzip_b64))
        print('Loading embedded user-supplied materialized data; no network required')
        return pd.read_csv(io.BytesIO(b),dtype={'source_id':str}),'embedded_user_results',hashlib.sha256(b).hexdigest()
    raise FileNotFoundError('Attach Twitter1516_Final_Ontology_Placebo_Results.zip or materialized_features.csv to Kaggle Input')


def audit_frame(df, minimum=100):
    required=['dataset','source_id','source_text','label']+NUMERIC_GRAPH
    missing=set(required)-set(df)
    if missing:raise ValueError('Missing required columns: '+str(sorted(missing)))
    df=df.copy()
    df['dataset']=df['dataset'].str.lower().str.strip()
    if not set(df.dataset.unique()) <= {'twitter15','twitter16'}:raise ValueError('Unexpected dataset names')
    if df.source_id.astype(str).str.contains(r'(?i)e\+\d+|\.0$').any():raise ValueError('Tweet IDs corrupted by numeric/float casting')
    if df[['dataset','source_id']].duplicated().any():raise ValueError('Duplicate dataset+source_id values')
    df['label']=pd.to_numeric(df['label'],errors='raise').astype(int)
    if not set(df.label.unique())=={0,1}:raise ValueError('Binary labels must be 0/1')
    df['source_text']=df['source_text'].fillna('').astype(str)
    for c in NUMERIC_GRAPH:
        df[c]=pd.to_numeric(df[c],errors='raise').astype(float)
        if not np.isfinite(df[c]).all():raise ValueError('NaN/Inf in graph column '+c)
    for ds,g in df.groupby('dataset'):
        if len(g)<minimum or g.label.nunique()!=2:raise ValueError('Insufficient data/classes in '+ds)
    df['group_text']=df.source_text.map(lambda x:hashlib.sha256(normalize(x).encode()).hexdigest())
    return df.reset_index(drop=True)


def eliminate_overlap(train,test,threshold=.975):
    """Inductive, label-blind exclusion: fit duplicate finder ONLY on source train."""
    train=train.copy().reset_index(drop=True);test=test.copy().reset_index(drop=True)
    shared_id=test.source_id.isin(set(train.source_id))
    shared_exact=test.group_text.isin(set(train.group_text))
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,max_features=65000,sublinear_tf=True)
    base=[normalize(s) for s in train.source_text]
    target=[normalize(s) for s in test.source_text]
    xs=vectorizer.fit_transform(base)
    xt=vectorizer.transform(target)
    similarities=cosine_similarity(xt,xs,dense_output=True).max(axis=1)
    near=similarities>=threshold
    exclude=np.asarray(shared_id|shared_exact|near)
    audit={'n_train':len(train),'n_target_before':len(test),'shared_id':int(shared_id.sum()),
           'shared_exact_text':int(shared_exact.sum()),'near_duplicate':int(near.sum()),
           'union_removed':int(exclude.sum()),'n_target_after':int((~exclude).sum()),
           'near_similarity_threshold':threshold}
    out=test.loc[~exclude].reset_index(drop=True)
    if len(out)<80 or out.label.nunique()!=2:raise ValueError('Too few independent test samples after duplicate removal')
    return train,out,audit


def lexical_features(texts):
    res=[]
    for s in texts:
        raw=str(s)
        t=raw.lower()
        n=max(1,len(raw))
        word_n=max(1,len(raw.split()))
        uncertainty=float(bool(re.search(r'\b(?:unconfirmed|unverified|rumou?r|unsubstantiated|not verified|fact.check|is it true)\b',t)))
        q=float('?' in raw)
        attrib=float(bool(re.search(r'\b(?:according to|sources? say|reportedly|reports? (?:that|say)|via @)\b',t)))
        hedging=float(bool(re.search(r'\b(?:may|might|alleged(?:ly)?|apparently|possibly|perhaps)\b',t)))
        res.append([uncertainty,q,attrib,hedging, float('http' in t or 'url' in t),
          float('@' in t),float('#' in t), sum(c.isupper() for c in raw)/n,math.log1p(word_n)])
    return np.asarray(res,dtype=np.float32)


def raw_graph_matrix(df):
    x=df[NUMERIC_GRAPH].to_numpy(dtype=np.float64).copy()
    for i,k in enumerate(NUMERIC_GRAPH):
        if any(w in k for w in ['nodes','edges','users','depth','branches','outdegree','count','duration','median','iqr']):
            x[:,i]=np.log1p(np.maximum(0,x[:,i]))
    return np.clip(x,-1e6,1e6).astype(np.float32)


class FrozenFuzzyOntology:
    """Unsupervised quantile-scale membership fit on source training data only.
    Uses *structural evidence*; DOES NOT infer truth, falsity or reply stance.
    """
    def fit(self, df):
        G=np.column_stack([df[c].to_numpy(float) for c in NUMERIC_GRAPH])
        self.ranges_={}
        for i,c in enumerate(NUMERIC_GRAPH):
            a=np.log1p(np.maximum(0,G[:,i])) if any(w in c for w in ['nodes','edges','users','depth','branches','outdegree','count','duration','median','iqr']) else G[:,i]
            q25,q50,q75=np.quantile(a,[.25,.50,.75])
            self.ranges_[c]=(float(q25),float(q50),float(q75))
        return self
    def _membership(self, name, v):
        q25,q50,q75=self.ranges_[name]
        if any(w in name for w in ['nodes','edges','users','depth','branches','outdegree','count','duration','median','iqr']):
            v=np.log1p(np.maximum(0,np.asarray(v,float)))
        spread=max((q75-q25)/2.0,0.10)
        return np.clip(0.5+(v-q50)/(4*spread),0,1)
    def transform(self,df,lex):
        col=lambda c:df[c].to_numpy(float)
        m=lambda c:self._membership(c,col(c))
        early_users=m('graph_early_60_unique_users')
        early_fraction=np.clip(col('graph_early_60_frac'),0,1)
        depth=m('graph_max_depth')
        branches=m('graph_root_branches')
        author=m('graph_unique_users')
        dur=m('graph_duration_min')
        late=m('graph_late_1440_count')
        leaf=np.clip(col('graph_leaf_frac'),0,1)
        coverage=np.clip(col('graph_coverage'),0,1)
        uncertainty=np.maximum(lex[:,0],np.minimum(1,0.45*lex[:,1]+0.6*lex[:,3]))
        attribution=np.clip(lex[:,2],0,1)  # A URL alone is NOT source attribution
        rapid=np.sqrt(early_users*early_fraction)
        persistence=np.sqrt(dur*late)
        multipath=np.sqrt(branches*depth)
        decentralized=branches*leaf
        depth_scale=max(1.0, np.expm1(self.ranges_['graph_mean_depth'][2]))
        deep=depth*np.sqrt(np.clip(col('graph_mean_depth')/depth_scale,0,1))
        independent=author*np.sqrt(np.clip(col('graph_unique_users')/(col('graph_nodes')+1e-6),0,1))
        uncearly=uncertainty*rapid
        sustained=depth*persistence
        review=1-(1-uncearly)*(1-multipath*0.55)*(1-persistence*0.35)
        reliability=np.sqrt(coverage)*(0.45+0.55*author)  # source-fitted membership avoids saturated 100% rule
        r=np.column_stack([rapid,persistence,multipath,decentralized,deep,independent,uncertainty,
                           attribution,uncearly,sustained,review,reliability]).astype(np.float32)
        # All source-domain fitted quantiles; no labels used.
        return np.clip(r,0,1)


def placebo_rules(onto, graph_nodes,seed):
    """Independent, conditional row-wise permutation preserving multivariate rules and size strata."""
    rng=np.random.default_rng(seed)
    n=len(onto)
    rank=pd.Series(np.asarray(graph_nodes).flatten()).rank(method='first')
    n_bins=min(5,max(1,n//50))
    bins=pd.qcut(rank,q=n_bins,labels=False,duplicates='drop').to_numpy()
    out=np.asarray(onto).copy()
    for b in np.unique(bins):
        ix=np.flatnonzero(bins==b)
        if len(ix)>1:out[ix]=onto[rng.permutation(ix)]
    return out


def catboost_model(cfg,seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=cfg.cat_iterations,depth=cfg.cat_depth,
      learning_rate=cfg.cat_learning_rate,l2_leaf_reg=6.,loss_function='Logloss',
      random_seed=int(seed),thread_count=4,verbose=False,allow_writing_files=False)


def fit_cat(x,y,cfg,seed):
    model=catboost_model(cfg,seed)
    model.fit(x,y,sample_weight=compute_sample_weight('balanced',y))
    return model


def sparse_text_features(train_text,test_text):
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=13000,
                        sublinear_tf=True,strip_accents='unicode',token_pattern=r'(?u)\b\w+\b')
    char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=22000,sublinear_tf=True)
    Xw=word.fit_transform(train_text); Yw=word.transform(test_text)
    Xc=char.fit_transform(train_text); Yc=char.transform(test_text)
    return hstack([Xw,Xc],format='csr'),hstack([Yw,Yc],format='csr')


def dense_text_features(xtr, xte, train_indices, test_indices, embeddings=None):
    # No PCA fit on held-out event or inner validation event.
    dim=min(48,xtr.shape[0]-2,xtr.shape[1]-1)
    if dim<2: raise ValueError('Not enough source rows for text compression')
    svd=TruncatedSVD(n_components=dim,random_state=SEED)
    Ztr=svd.fit_transform(xtr).astype(np.float32)
    Zte=svd.transform(xte).astype(np.float32)
    if embeddings is not None:
        etr=embeddings[train_indices];ete=embeddings[test_indices]
        pc=PCA(n_components=min(48,etr.shape[1],len(etr)-1),random_state=SEED)
        Ztr=np.hstack((Ztr,pc.fit_transform(etr).astype(np.float32)))
        Zte=np.hstack((Zte,pc.transform(ete).astype(np.float32)))
    return Ztr,Zte


def encode_embeddings(df,cfg,outdir):
    if not cfg.use_minilm:return None,'disabled_by_config'
    data_key=hashlib.sha256(('\n'.join(df.dataset+'|'+df.source_id+'|'+df.source_text)).encode('utf8')).hexdigest()[:16]
    model_tag=re.sub('[^a-z0-9]+','_',cfg.minilm_model.lower())
    path=outdir/f'embeddings_{model_tag}_{data_key}.npy'
    if path.exists():
        arr=np.load(path)
        if arr.shape[0]==len(df):print('Reusing frozen pretrained embeddings');return arr,'cache'
    try:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(cfg.minilm_model)
        arr=np.asarray(model.encode(df.source_text.tolist(),batch_size=cfg.minilm_batch_size,
            show_progress_bar=True,normalize_embeddings=True),dtype=np.float32)
        np.save(path,arr)
        return arr,'pretrained_frozen'
    except Exception as e:
        if cfg.allow_fallback_encoder:
            warnings.warn('MiniLM unavailable; results explicitly marked as TF-IDF only: '+str(e))
            return None,'fallback_'+type(e).__name__
        raise RuntimeError('MiniLM was required but could not be loaded; enable Kaggle Internet or set allow_fallback_encoder=True explicitly') from e


def _join_numeric(ztext,graph,lex,onto=None,classic=None):
    parts=[ztext,graph,lex]
    if onto is not None:parts.append(onto)
    if classic is not None:parts.append(classic)
    x=np.hstack(parts).astype(np.float32)
    return np.nan_to_num(x,nan=0,posinf=1e6,neginf=-1e6)


def feature_bundle(train,test,embeddings,train_index,test_index):
    xs,xt=sparse_text_features(train.source_text.tolist(),test.source_text.tolist())
    Zs,Zt=dense_text_features(xs,xt,train_index,test_index,embeddings)
    graph_s,graph_t=raw_graph_matrix(train),raw_graph_matrix(test)
    lex_s,lex_t=lexical_features(train.source_text),lexical_features(test.source_text)
    onto=FrozenFuzzyOntology().fit(train)
    ont_s,ont_t=onto.transform(train,lex_s),onto.transform(test,lex_t)
    classics=np.column_stack([train[c].to_numpy(float) if c in train else np.zeros(len(train)) for c in CLASSIC_RULES]).astype(np.float32)
    classict=np.column_stack([test[c].to_numpy(float) if c in test else np.zeros(len(test)) for c in CLASSIC_RULES]).astype(np.float32)
    X_B=_join_numeric(Zs,graph_s,np.zeros((len(train),0),np.float32))
    T_B=_join_numeric(Zt,graph_t,np.zeros((len(test),0),np.float32))
    X_C=_join_numeric(Zs,graph_s,lex_s)
    T_C=_join_numeric(Zt,graph_t,lex_t)
    X_D=_join_numeric(Zs,graph_s,lex_s,ont_s)
    T_D=_join_numeric(Zt,graph_t,lex_t,ont_t)
    X_F=_join_numeric(Zs,graph_s,lex_s,None, classics)
    T_F=_join_numeric(Zt,graph_t,lex_t,None, classict)
    return (xs,xt),{'B_TextGraph':(X_B,T_B),'C_RawEvidence':(X_C,T_C),
           'D_FuzzyOntology':(X_D,T_D),'F_LegacyRules':(X_F,T_F)},(ont_s,ont_t), (Zs,Zt,graph_s,graph_t,lex_s,lex_t)


def oof_source(train,src_indices,embeddings,cfg):
    n=len(train);y=train.label.to_numpy()
    fold=min(cfg.cv_folds,np.bincount(y).min())
    if fold<2:raise ValueError('Too few examples per class')
    skf=StratifiedGroupKFold(n_splits=fold,shuffle=True,random_state=cfg.seed)
    oof={k:np.full(n,np.nan) for k in ['A_Text','B_TextGraph','C_RawEvidence','D_FuzzyOntology','F_LegacyRules']}
    for fi,(a,b) in enumerate(skf.split(train.source_text,y,groups=train.group_text)):
        tr=train.iloc[a].reset_index(drop=True)
        te=train.iloc[b].reset_index(drop=True)
        (xs,xt),blocks,_,_=feature_bundle(tr,te,embeddings,src_indices[a],src_indices[b])
        lr=LogisticRegression(C=.6,class_weight='balanced',max_iter=500,solver='liblinear',random_state=cfg.seed)
        lr.fit(xs,y[a]);oof['A_Text'][b]=lr.predict_proba(xt)[:,1]
        for key in ['B_TextGraph','C_RawEvidence','D_FuzzyOntology','F_LegacyRules']:
            X,T=blocks[key]
            m=fit_cat(X,y[a],cfg,cfg.seed+fi)
            oof[key][b]=m.predict_proba(T)[:,1]
        print('Source-domain OOF fold',fi+1,'/',fold,'done')
    for k,v in oof.items():
        if not np.isfinite(v).all():raise AssertionError('Missing OOF values for '+k)
    return oof


def f1macro(y,p,thr):return float(f1_score(y,(p>=thr).astype(int),average='macro',zero_division=0))


def select_protocol(y,oof):
    """Alpha, thresholds and configuration selection use ONLY source-domain OOF labels."""
    params={}
    bas=oof['A_Text']
    # alpha fraction for numeric branch, trained same way for all configurations
    for conf in ['B_TextGraph','C_RawEvidence','D_FuzzyOntology','F_LegacyRules']:
        best=(-1,None,None)
        for alpha in [.35,.5,.65,.8,1.0]:
            p=alpha*oof[conf]+(1-alpha)*bas
            for thr in np.linspace(.35,.65,13):
                m=f1macro(y,p,thr)
                # Tie deterministic: choose alpha closer to .65 and thr closer to .5
                score=(m,-abs(alpha-.65),-abs(thr-.5))
                if best[1] is None or score>best[0]:best=(score,alpha,thr)
        params[conf]={'alpha':float(best[1]),'threshold':float(best[2]),'oof_macro_f1':float(best[0][0])}
    # One inner-selected model, not selected by target performance.
    winner=max(['C_RawEvidence','D_FuzzyOntology'],key=lambda k:(params[k]['oof_macro_f1'],-(k=='D_FuzzyOntology')))
    params['G_TrainSelected']={'source_cv_winner':winner}
    return params


def metric_row(y,p,threshold):
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'rumor_f1':float(f1_score(y,pred,pos_label=1,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'auprc':float(average_precision_score(y,p)),
            'mcc':float(matthews_corrcoef(y,pred)),
            'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn)}


def bootstrap_paired(y,p1,p0,t1,t0,n_boot=1000,seed=42):
    rng=np.random.default_rng(seed);n=len(y);delta=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        if len(np.unique(y[idx]))<2:continue
        delta.append(f1macro(y[idx],p1[idx],t1)-f1macro(y[idx],p0[idx],t0))
    return {'n_boot':len(delta),'delta_ci_low':float(np.quantile(delta,.025)),
            'delta_ci_high':float(np.quantile(delta,.975)),
            'delta_macro_f1':f1macro(y,p1,t1)-f1macro(y,p0,t0)}


def write_rdf(outdir,df,rules,limit=12):
    try:
        from rdflib import Graph,Namespace,URIRef,Literal
        from rdflib.namespace import RDF, XSD
        g=Graph(); ex=Namespace('https://example.org/rumor-evidence#');g.bind('evidence',ex)
        for i in range(min(len(df),limit)):
            item=df.iloc[i]
            node=URIRef(str(ex)+'claim/'+str(item.dataset)+'/'+str(item.source_id))
            g.add((node,RDF.type,ex.SourceClaim))
            for k,v in zip(RULE_KEYS,rules[i]):
                evidence=URIRef(str(ex)+'observation/'+str(item.dataset)+'/'+str(item.source_id)+'/'+k)
                g.add((evidence,RDF.type,ex.SoftRuleEvidence))
                g.add((evidence,ex.ruleIdentifier,Literal(k)))
                g.add((evidence,ex.membershipScore,Literal(round(float(v),5),datatype=XSD.float)))
                g.add((evidence,ex.derivedFrom,node))
                g.add((node,ex.hasEvidence,evidence))
        g.serialize(destination=str(outdir/('ontology_provenance_'+str(df.iloc[0].dataset)+'.ttl')),format='turtle')
    except ImportError: (outdir/('ontology_provenance_'+str(df.iloc[0].dataset)+'.ttl')).write_text('# rdflib not available\n')


def analyze_one_direction(df,source,target,embeddings,cfg,output_dir):
    src=df.loc[df.dataset==source].copy().reset_index(drop=True)
    tgt=df.loc[df.dataset==target].copy().reset_index(drop=True)
    # Keep global embedding mapping stable (IDs are strings, no loss of precision)
    src['_embedding_idx']=df.index[df.dataset==source].to_numpy()
    tgt['_embedding_idx']=df.index[df.dataset==target].to_numpy()
    src,tgt,audit=eliminate_overlap(src,tgt,cfg.near_duplicate_threshold)
    audit.update(direction=f'{source}_to_{target}')
    print('Transfer',audit['direction'],'n_train',len(src),'n_test',len(tgt),'overlap removed',audit['union_removed'])
    source_indices=src['_embedding_idx'].to_numpy(dtype=int)
    target_indices=tgt['_embedding_idx'].to_numpy(dtype=int)
    y=src.label.to_numpy(int);yt=tgt.label.to_numpy(int)
    oof=oof_source(src,source_indices,embeddings,cfg)
    params=select_protocol(y,oof)
    print('Source-only model selection:',params)
    (xs,xt),blocks,(rule_train,rule_test),extras=feature_bundle(src,tgt,embeddings,source_indices,target_indices)
    lr=LogisticRegression(C=.6,class_weight='balanced',max_iter=500,solver='liblinear',random_state=cfg.seed)
    lr.fit(xs,y);ptext=lr.predict_proba(xt)[:,1]
    preds={'A_Text':ptext};thresholds={'A_Text':.5};models={}
    numeric_predictions={}
    for key in ['B_TextGraph','C_RawEvidence','D_FuzzyOntology','F_LegacyRules']:
        X,T=blocks[key]
        m=fit_cat(X,y,cfg,cfg.seed);models[key]=m
        numeric_predictions[key]=m.predict_proba(T)[:,1]
        alpha=params[key]['alpha']
        preds[key]=(1-alpha)*ptext+alpha*numeric_predictions[key]
        thresholds[key]=params[key]['threshold']
    # Primary controlled D vs C: identical source-selected alpha and threshold.
    # Otherwise a gain can come from threshold tuning, not Ontology rules.
    alpha_c=params['C_RawEvidence']['alpha']
    threshold_c=params['C_RawEvidence']['threshold']
    preds['I_SourceTunedOntology']=preds['D_FuzzyOntology'].copy()
    thresholds['I_SourceTunedOntology']=thresholds['D_FuzzyOntology']
    preds['D_FuzzyOntology']=(1-alpha_c)*ptext+alpha_c*numeric_predictions['D_FuzzyOntology']
    thresholds['D_FuzzyOntology']=threshold_c
    chosen=params['G_TrainSelected']['source_cv_winner']
    selected_key='I_SourceTunedOntology' if chosen=='D_FuzzyOntology' else 'C_RawEvidence'
    preds['G_TrainSelected']=preds[selected_key]
    thresholds['G_TrainSelected']=thresholds[selected_key]
    importance=[]
    imp=models['D_FuzzyOntology'].get_feature_importance()
    for idx,key in enumerate(RULE_KEYS):
        importance.append({'direction':audit['direction'],'rule':key,'importance':float(imp[-len(RULE_KEYS)+idx])})
    # Test-only shuffle diagnostic: NOT used to select models.
    perturbed=placebo_rules(rule_test,tgt.graph_nodes.to_numpy(),cfg.seed+7001)
    D_X,D_T=blocks['D_FuzzyOntology']
    shuffled_T=D_T.copy();shuffled_T[:,-len(RULE_KEYS):]=perturbed
    p_d_shuffled=(1-alpha_c)*ptext+alpha_c*models['D_FuzzyOntology'].predict_proba(shuffled_T)[:,1]
    sensitivity={'direction':audit['direction'],
        'mean_absolute_probability_change':float(np.mean(np.abs(p_d_shuffled-preds['D_FuzzyOntology']))),
        'macro_f1_after_shuffling':f1macro(yt,p_d_shuffled,thresholds['D_FuzzyOntology']),
        'macro_f1_original':f1macro(yt,preds['D_FuzzyOntology'],thresholds['D_FuzzyOntology'])}
    # Placebo: fixed alpha/threshold and same estimator, conditional permutation + refit each repeat.
    # Shares non-rule model and source-trained text branch, does NOT permute target labels.
    ztr,zte,graph_tr,graph_te,lex_tr,lex_te=extras
    placebo=[]
    for j in range(cfg.placebo_repeats):
        rtrain=placebo_rules(rule_train,src.graph_nodes.to_numpy(),cfg.seed+1000+j*3)
        rtest=placebo_rules(rule_test,tgt.graph_nodes.to_numpy(),cfg.seed+2000+j*3)
        X=_join_numeric(ztr,graph_tr,lex_tr,rtrain)
        T=_join_numeric(zte,graph_te,lex_te,rtest)
        pm=fit_cat(X,y,cfg,cfg.seed)
        q=(1-alpha_c)*ptext+alpha_c*pm.predict_proba(T)[:,1]
        placebo.append(q)
    rows=[]; predrows=[]
    for key,p in preds.items():
        metric=metric_row(yt,p,thresholds[key]);metric.update(direction=audit['direction'],config=key,
                                                   replicate=-1,n_train=len(src),n_test=len(tgt))
        rows.append(metric)
        predrows.extend({'direction':audit['direction'],'config':key,'replicate':-1,'source_id':sid,
                         'y_true':int(yy),'p_rumor':float(pp),'y_pred':int(pp>=thresholds[key])}
                        for sid,yy,pp in zip(tgt.source_id,yt,p))
    for j,q in enumerate(placebo):
        th=thresholds['D_FuzzyOntology']
        mr=metric_row(yt,q,th);mr.update(direction=audit['direction'],config='E_ConditionalPlacebo',
                                        replicate=j,n_train=len(src),n_test=len(tgt))
        rows.append(mr)
        predrows.extend({'direction':audit['direction'],'config':'E_ConditionalPlacebo','replicate':j,
                         'source_id':sid,'y_true':int(yy),'p_rumor':float(pp),'y_pred':int(pp>=th)}
                        for sid,yy,pp in zip(tgt.source_id,yt,q))
    changes={}
    for ref in ['C_RawEvidence','F_LegacyRules']:
        p0=preds[ref]
        t0=thresholds[ref]
        changes['D_minus_'+ref]=bootstrap_paired(yt,preds['D_FuzzyOntology'],p0,
                                         thresholds['D_FuzzyOntology'],t0,seed=cfg.seed)
    # Evaluating the *mean* placebo probabilities is diagnostic, not the placebo replicate mean F1.
    deltas=[]
    dmetric=metric_row(yt,preds['D_FuzzyOntology'],thresholds['D_FuzzyOntology'])
    for j,q in enumerate(placebo):
        pm=metric_row(yt,q,thresholds['D_FuzzyOntology'])
        deltas.append(dmetric['macro_f1']-pm['macro_f1'])
    rulecov={'direction':audit['direction'],**{name:float((rule_train[:,i]>.5).mean()) for i,name in enumerate(RULE_KEYS)}}
    out={'metrics':rows,'predictions':predrows,'overlap':audit,'selection':{'direction':audit['direction'],**params},
         'contrasts':{'direction':audit['direction'],**changes,
                 'D_minus_placebo_f1_replicates':deltas,
                 'conditional_randomization_p_upper_bound':float((1+sum(x<=0 for x in deltas))/(len(deltas)+1)),
                 'placebo_probability_shift_mean':float(np.mean(np.abs(np.mean(placebo,axis=0)-preds['D_FuzzyOntology'])))},
         'rule_coverage':rulecov,'rule_importance':importance,'rule_sensitivity':sensitivity,
         'primary_contrast_fixed_threshold':True}
    return out, (src,rule_train)


def run(cfg:Config,embedded_gzip_b64=None):
    start=time.time()
    outdir=Path(cfg.workdir)/cfg.output_prefix
    outdir.mkdir(parents=True,exist_ok=True)
    df,origin,data_sha=read_input_materialized(cfg.input_root,cfg.workdir,embedded_gzip_b64)
    df=audit_frame(df,cfg.min_examples)
    df.to_csv(outdir/'input_manifest_rows.csv',index=False)
    embeddings,emb_status=encode_embeddings(df,cfg,outdir)
    manifests={'data_origin':origin,'data_sha256':data_sha,'full_benchmark':False,
               'n_total':len(df),'dataset_counts':df.dataset.value_counts().to_dict(),
               'encoding_status':emb_status,'config':asdict(cfg),
               'note':'This is a development follow-up on a previously inspected Twitter15/16 subset. Not independent held-out final confirmation.',
               'method_note':'Fuzzy structural ontology memberships are derived from graph/source features and do not infer fact truth or support/deny stance.'}
    (outdir/'run_manifest.json').write_text(json.dumps(manifests,indent=2,ensure_ascii=False),encoding='utf8')
    allres=[]
    for src,tgt in [('twitter15','twitter16'),('twitter16','twitter15')]:
        res,rule_data=analyze_one_direction(df,src,tgt,embeddings,cfg,outdir)
        allres.append(res)
        write_rdf(outdir,*rule_data)
        # Persist every complete direction for safe recovery; never reuse incomplete files as completed folds.
        (outdir/f'done_{src}_to_{tgt}.json').write_text(json.dumps(res,ensure_ascii=False))
    metrics=pd.DataFrame([r for v in allres for r in v['metrics']])
    preds=pd.DataFrame([r for v in allres for r in v['predictions']])
    metric_columns=['macro_f1','recall','precision','auprc','mcc','FN','FP','accuracy']
    summary=metrics.groupby(['direction','config'],as_index=False)[metric_columns].mean()
    overlap=pd.DataFrame([x['overlap'] for x in allres])
    selection=pd.json_normalize([x['selection'] for x in allres])
    coverage=pd.DataFrame([x['rule_coverage'] for x in allres])
    importance=pd.DataFrame([x for v in allres for x in v['rule_importance']])
    sensitivity=pd.DataFrame([x['rule_sensitivity'] for x in allres])
    contrasts=pd.DataFrame([{'direction':r['contrasts']['direction'],
        'comparison':k,**v} for r in allres for k,v in r['contrasts'].items() if isinstance(v,dict)])
    placebo_test=pd.DataFrame([{'direction':r['contrasts']['direction'],
        'delta_real_minus_placebo_mean':float(np.mean(r['contrasts']['D_minus_placebo_f1_replicates'])),
        'delta_real_minus_placebo_sd':float(np.std(r['contrasts']['D_minus_placebo_f1_replicates'],ddof=1)) if len(r['contrasts']['D_minus_placebo_f1_replicates'])>1 else 0.,
        'randomization_p_upper_bound':r['contrasts']['conditional_randomization_p_upper_bound'],
        'n_placebo':len(r['contrasts']['D_minus_placebo_f1_replicates'])} for r in allres])
    for p,frame in [('metrics.csv',metrics),('summary.csv',summary),('predictions.csv',preds),
         ('overlap_audit.csv',overlap),('source_only_cv_selection.csv',selection),
         ('ontology_rule_coverage.csv',coverage),('ontology_rule_importance.csv',importance),
         ('ontology_shuffling_diagnostics.csv',sensitivity),('paired_bootstrap.csv',contrasts),
         ('placebo_test.csv',placebo_test)]:
        frame.to_csv(outdir/p,index=False)
    rules=[{'name':k,'definition':'Source-only or graph-based fuzzy evidence; see pipeline for exact membership',
            'not_proof_of_truth':True} for k in RULE_KEYS]
    pd.DataFrame(rules).to_csv(outdir/'ontology_rule_catalog.csv',index=False)
    (outdir/'README_RESULTS.md').write_text('''# Twitter15/16 v17 single-run experiment\n\n'
    'Development experiment — not untouched confirmatory evidence. The sample comes from a previously inspected '
    'Twitter15/16 subset, not the complete original benchmark. Source labels only for fitting/validation.\n\n'
    'D vs C: fuzzy rules beyond text+graph+raw cues; D vs E: matched conditional row-shuffled rules, '
    'refitting all placebo models. Soft rules are inductive biases, not independent information or OWL-DL reasoning.\n\n'
    'A pretrained MiniLM encoder is frozen and cannot use target labels. See run_manifest for status.\n'
    'The sample size and duplicate-exclusion audit must accompany all scientific claims.\n''')
    (outdir/'report_summary.md').write_text('# Completed one-run results\n\n'+summary.round(4).to_markdown(index=False)+
      '\n\n## Overlap exclusions\n\n'+overlap.to_markdown(index=False)+'\n')
    zip_path=Path(cfg.workdir)/(cfg.output_prefix+'.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in outdir.rglob('*'):
            if f.is_file() and not f.name.startswith('embeddings_'):
                z.write(f,f.relative_to(outdir))
    print('Complete in %.1f seconds'%(time.time()-start));print('Results:',zip_path)
    print(summary[['direction','config','macro_f1','recall','FN','FP']].round(4).to_string(index=False))
    return zip_path
