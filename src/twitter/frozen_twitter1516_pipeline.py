"""Reproducible Twitter15/16 full-source, tree, temporal/ontology ablation study.
The original gszswork mirror lacks source_tweets.txt; source text is fetched from a
second publicly available mirror, keyed by the original integer-string tweet IDs.
No test labels are used to choose features, parameters, or thresholds.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, re, shutil, time, zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, average_precision_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ORIGINAL_SHA='6a33ebd91d4cafeb5a42eabc5727226fec7bb0e5'
SOURCE_SHA='fc2d35e67d795feab30d22705ae3395eb97b9445'
ORIGINAL_ZIP=f'https://codeload.github.com/gszswork/Twitter15_16_dataset/zip/{ORIGINAL_SHA}'
SOURCE_URL=f'https://raw.githubusercontent.com/leekhithnunna/Twitter_15_16_Rumour_Verification/{SOURCE_SHA}/{{dataset}}/source_tweets.txt'
EXPECTED={'twitter15':1490,'twitter16':818}
LABEL={'non-rumor':0,'false':1,'true':1,'unverified':1}
TIME_WINDOWS=[60,360,1440,'full']
G_COLS=['g_log_nodes','g_log_edges','g_log_users','g_user_ratio','g_root_degree','g_max_depth','g_mean_depth','g_leaf_ratio','g_breadth','g_repeated_author','g_depth2_ratio']
R_COLS=['e_log_duration','e_log_rate','e_first_delay','e_median_delay','e_p90_delay','e_frac_1h','e_frac_6h','e_burst_early','e_parent_lag','e_log_independent_branches']
O_COLS=['o_rapid_independent_burst','o_deep_persistence','o_broad_amplification','o_delayed_revival','o_multi_branch_diffusion','o_temporal_concentration']


def fetch(url: str, path:Path, timeout=90):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=2,status_forcelist=[429,500,502,503,504])))
    r=session.get(url,timeout=timeout);r.raise_for_status()
    if len(r.content)<100:raise RuntimeError(f'Unexpected short download {url}')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
    return {'url':url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()}


def prepare_data(root:Path, input_root:Path|None=None, allow_download=True):
    """Loads a Kaggle-mounted full original archive or downloads pinned repository.
    A source-text mirror is required and joined by tweet ID, never synthesized.
    """
    root.mkdir(parents=True,exist_ok=True); raw=root/'raw'; raw.mkdir(exist_ok=True)
    manifests=[]
    repo=next((p for p in raw.rglob('twitter15/label.txt') if (p.parent.parent/'twitter16/label.txt').exists()),None)
    if repo is None:
        candidates=[]
        if input_root and input_root.exists():
            candidates=list(input_root.rglob('label.txt'))
        for p in candidates:
            if p.parent.name=='twitter15' and (p.parent.parent/'twitter16/label.txt').exists():
                repo=p;break
    if repo is None:
        zipped=raw/'twitter_original_full.zip'
        if input_root and input_root.exists():
            matches=[p for p in input_root.rglob('*.zip') if ('twitter' in p.name.lower() and ('15' in p.name or 'detect' in p.name.lower()))]
            if matches:zipped=matches[0]
        if not zipped.exists():
            if not allow_download:raise FileNotFoundError('Original repository absent; enable Internet or add the full GitHub ZIP to Kaggle Input.')
            try:
                manifests.append(fetch(ORIGINAL_ZIP,zipped,timeout=150))
            except Exception as e:
                alternate=f'https://github.com/gszswork/Twitter15_16_dataset/archive/{ORIGINAL_SHA}.zip'
                print('codeload failed; trying GitHub archive URL:',str(e)[:120],flush=True)
                manifests.append(fetch(alternate,zipped,timeout=150))
        with zipfile.ZipFile(zipped) as z:
            if any('..' in Path(n).parts or n.startswith('/') for n in z.namelist()):raise ValueError('Unsafe ZIP paths')
            z.extractall(raw)
        repo=next((p for p in raw.rglob('twitter15/label.txt') if (p.parent.parent/'twitter16/label.txt').exists()),None)
    if repo is None:raise FileNotFoundError('Cannot find twitter15/label.txt and twitter16/label.txt together')
    base=repo.parent.parent
    for ds in EXPECTED:
        fp=base/ds/'source_tweets.txt'
        if not fp.exists():
            if not allow_download:raise FileNotFoundError(f'Missing {fp}. This GitHub mirror provides labels and trees but not source texts.')
            manifests.append(fetch(SOURCE_URL.format(dataset=ds),fp))
    (root/'source_download_manifest.json').write_text(json.dumps({'repo_sha':ORIGINAL_SHA,'source_mirror_sha':SOURCE_SHA,'files':manifests,'base':str(base)},indent=2),encoding='utf-8')
    return base


def norm_text(s:str):
    import unicodedata
    s=unicodedata.normalize('NFKC',str(s)).lower()
    s=re.sub(r'https?://\S+|\bwww\.\S+',' URL ',s)
    s=re.sub(r'@[\w_]+',' USER ',s)
    return re.sub(r'[^\w]+',' ',s,flags=re.UNICODE).strip()


def parse_node(token):
    """Returns (user_id, tweet_id, delay_min) without floating Twitter ID conversion."""
    a=ast.literal_eval(token.strip())
    if not isinstance(a,list) or len(a)!=3:raise ValueError('bad node literal')
    user,tw,delay=[str(x) for x in a]
    delay=float(delay)
    if not np.isfinite(delay):raise ValueError('Nonfinite time')
    return (user,tw,delay)


def parse_tree(path:Path):
    nodes=set();edges=set();anomaly=Counter();bad_lines=[]
    for lineno,line in enumerate(path.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
        line=line.strip()
        if not line:continue
        try:
            left,right=line.split('->',1)
            u,v=parse_node(left),parse_node(right)
            if u==v:anomaly['self_edges']+=1;continue
            if (u,v) in edges:anomaly['duplicate_edges']+=1
            edges.add((u,v));nodes.update([u,v])
            if v[2]<0 or u[2]<0:anomaly['negative_delay']+=1
            if v[2]<u[2]-1e-8:anomaly['backwards_edge']+=1
        except Exception as ex:
            anomaly['malformed_lines']+=1
            if len(bad_lines)<5:bad_lines.append(f'{lineno}: {type(ex).__name__}: {line[:130]}')
    return {'nodes':nodes,'edges':edges,'issues':dict(anomaly),'examples':bad_lines}


def load_datasets(base:Path, strict=True):
    out=[];tree_map={};rows=[]
    for ds,exp in EXPECTED.items():
        lab_path=base/ds/'label.txt';text_path=base/ds/'source_tweets.txt';tree_dir=base/ds/'tree'
        labels={}; label_dup=[]
        for l in lab_path.read_text(encoding='utf-8-sig').splitlines():
            if ':' not in l:continue
            label,t_id=l.strip().split(':',1);t_id=t_id.strip()
            if t_id in labels:label_dup.append(t_id)
            labels[t_id]=label.strip().lower()
        texts={};text_dup=[]
        for l in text_path.read_text(encoding='utf-8-sig',errors='replace').splitlines():
            if '\t' not in l:continue
            t_id,tx=l.split('\t',1);t_id=t_id.strip()
            if t_id in texts:text_dup.append(t_id)
            texts[t_id]=tx.strip()
        fids={x.stem for x in tree_dir.glob('*.txt')}
        for t_id,label in labels.items():
            out.append({'dataset':ds,'id':t_id,'label_raw':label,'label':LABEL.get(label,np.nan),'source_text':texts.get(t_id,''),'normalized':norm_text(texts.get(t_id,'')),'has_tree':t_id in fids})
        rows.append({'dataset':ds,'expected':exp,'label_rows':len(labels),'source_rows':len(texts),'tree_files':len(fids),'duplicate_label_ids':len(label_dup),'duplicate_source_ids':len(text_dup),'missing_source':len(set(labels)-set(texts)),'missing_tree':len(set(labels)-fids),'unlabelled_trees':len(fids-set(labels)),'orphan_source':len(set(texts)-set(labels))})
        if strict and (len(labels)!=exp or len(texts)!=exp or len(fids)!=exp or set(labels)!=set(texts) or set(labels)!=fids or label_dup or text_dup or any(v not in LABEL for v in labels.values()) or any(not v.strip() for v in texts.values())):
            raise RuntimeError(f'{ds}: incomplete or mismatched source/label/tree inventory: {rows[-1]}')
    return pd.DataFrame(out),pd.DataFrame(rows)


def duplicate_audit(frame:pd.DataFrame,root:Path):
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for (d,tx),gr in frame.groupby(['dataset','normalized'],dropna=False):
        if tx and len(gr)>1:
            rows.append({'scope':d,'kind':'exact_normalized','normalized_preview':str(tx)[:160],'n':len(gr),'ids':'|'.join(gr.id.tolist())})
    cross=[]
    for t,gr in frame.groupby('normalized'):
        if t and gr.dataset.nunique()>1:
            cross.append({'kind':'cross_exact_text','n':len(gr),'ids':'|'.join(gr.dataset+'::'+gr.id),'preview':t[:180]})
    a=frame[frame.dataset=='twitter15'];b=frame[frame.dataset=='twitter16']
    idset=set(a.id)&set(b.id)
    label_conflicts=[]
    for normalized,group in frame.groupby('normalized'):
        if normalized and group.label_raw.nunique()>1:
            label_conflicts.append({'normalized_preview':normalized[:200],
                    'different_labels':';'.join(sorted(group.label_raw.unique())),
                    'ids':'|'.join(group.dataset+'::'+group.id)})
    pd.DataFrame(label_conflicts,columns=['normalized_preview','different_labels','ids']).to_csv(root/'duplicate_text_label_conflicts.csv',index=False)
    # Sparse character similarities. Never inspect labels to decide overlap.
    from sklearn.metrics.pairwise import cosine_similarity
    v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,sublinear_tf=True,max_features=110000)
    x=v.fit_transform(frame.normalized.fillna('').tolist());
    xa=x[frame.index.isin(a.index)];xb=x[frame.index.isin(b.index)]
    near=[]; within_near=[]; blocking=set(); threshold=.93
    # At most ~1.2 million sparse pair comparisons, batched for RAM safety.
    for i in range(0,xa.shape[0],96):
        sim=(xa[i:i+96]@xb.T).tocoo()
        for j,k,s in zip(sim.row,sim.col,sim.data):
            p=a.iloc[i+j];q=b.iloc[k]
            if s < threshold or p.id==q.id or p.normalized==q.normalized:continue
            if len(p.normalized)<22 or len(q.normalized)<22:continue
            near.append({'twitter15_id':p.id,'twitter16_id':q.id,'cosine_char_3to5':round(float(s),6)})
            blocking.add(q.id)
    # Near duplicates inside each dataset, excluding exact normalized duplicates.
    for ds,indices in [('twitter15',np.where(frame.dataset.to_numpy()=='twitter15')[0]),('twitter16',np.where(frame.dataset.to_numpy()=='twitter16')[0])]:
        xx=x[indices]
        for i in range(0,xx.shape[0],64):
            nearblock=(xx[i:i+64] @ xx.T).tocoo()
            for j,k,sim in zip(nearblock.row,nearblock.col,nearblock.data):
                ix=i+j
                if ix>=k or sim < threshold: continue
                p=frame.iloc[indices[ix]];q=frame.iloc[indices[k]]
                if p.normalized==q.normalized or len(p.normalized)<22 or len(q.normalized)<22:continue
                within_near.append({'dataset':ds,'id_a':p.id,'id_b':q.id,'cosine_char_3to5':round(float(sim),6)})
    pd.DataFrame(within_near,columns=['dataset','id_a','id_b','cosine_char_3to5']).to_csv(root/'within_dataset_near_pairs.csv',index=False)
    pd.DataFrame(rows,columns=['scope','kind','normalized_preview','n','ids']).to_csv(root/'within_dataset_duplicate_groups.csv',index=False)
    pd.DataFrame(cross,columns=['kind','n','ids','preview']).to_csv(root/'cross_dataset_exact_groups.csv',index=False)
    pd.DataFrame(near,columns=['twitter15_id','twitter16_id','cosine_char_3to5']).to_csv(root/'cross_dataset_near_pairs.csv',index=False)
    match15=frame[frame.dataset=='twitter15'];match16=frame[frame.dataset=='twitter16']
    shared_exact_text=set(match15.normalized)&set(match16.normalized)
    summary={'same_source_id_between_datasets':len(idset),'cross_exact_normalized_groups':len(cross),'within_duplicate_groups':len(rows),'cross_near_duplicate_pairs_excluding_exact':len(near),'within_near_duplicate_pairs_excluding_exact':len(within_near),'duplicate_text_groups_with_disagreeing_labels':len(label_conflicts),'near_pairs_threshold':threshold,'empty_source':int((frame.normalized=='').sum()),'label_distribution':frame.groupby(['dataset','label_raw']).size().to_dict().__str__()}
    (root/'text_audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    # Holdout blocking for either direction, intentionally symmetric.
    near15={r['twitter15_id'] for r in near};near16={r['twitter16_id'] for r in near}
    blocks={'twitter15':set(idset)|set(match15[match15.normalized.isin(shared_exact_text)].id)|near15,'twitter16':set(idset)|set(match16[match16.normalized.isin(shared_exact_text)].id)|near16}
    return summary,blocks


def gini(arr):
    a=np.asarray(arr,dtype=float)
    if not len(a) or a.sum()<=0:return 0.
    a=np.sort(a);n=len(a)
    return float((2*np.dot(np.arange(1,n+1),a)/(n*a.sum()))-(n+1)/n)


def snapshot_features(tree,window):
    n={x for x in tree['nodes'] if (window=='full' or x[2]<=float(window)) and x[2]>=0}
    # Represent the synthetic ROOT even when a tree is empty.
    edges=[(p,c) for p,c in tree['edges'] if p in n and c in n and c[2]>=p[2]]
    node_real=[v for v in n if v[1]!='ROOT']
    parent={c:p for p,c in edges}
    children=defaultdict(list)
    for u,v in edges:children[u].append(v)
    starts=[v for v in node_real if v not in parent or parent[v][1]=='ROOT']
    # Use a bounded iterative breadth-first traversal, no future nodes.
    depth={v:0 for v in starts}; dq=deque(starts)
    while dq:
        v=dq.popleft()
        for c in children[v]:
            dd=depth[v]+1
            if dd>len(n):continue
            if c not in depth or depth[c]>dd:depth[c]=dd;dq.append(c)
    times=sorted(v[2] for v in node_real if v[2]>0)
    dvalues=[depth.get(v,0) for v in node_real]
    nonroot=[v for v in node_real if v[2]>0]
    uid=[v[0] for v in node_real]
    counts=Counter(uid);nc=len(node_real);nun=len(counts)
    counts_array=np.asarray(list(counts.values()),dtype=float)
    ent=float(-sum((x/nc)*math.log(x/nc) for x in counts_array)) if nc else 0
    ent=ent/math.log(nun) if nun>1 else 0
    root_degree=sum(len(children[v]) for v in starts)
    outdeg=[len(children[v]) for v in node_real]
    span=max(times)-min(times) if len(times)>1 else 0.
    med=float(np.median(times)) if times else 0.
    p90=float(np.quantile(times,.9)) if times else 0.
    tlist=np.asarray(times,dtype=float)
    early=int(np.sum(tlist<=60))
    firstroot=sum(1 for v in starts if v[2]<=60)
    f={
     'g_log_nodes':np.log1p(nc),'g_log_edges':np.log1p(len(edges)),'g_log_users':np.log1p(nun),
     'g_user_ratio':nun/max(nc,1),'g_root_degree':np.log1p(root_degree),
     'g_max_depth':np.log1p(max(dvalues,default=0)),'g_mean_depth':np.mean(dvalues) if dvalues else 0.,
     'g_leaf_ratio':np.mean([v==0 for v in outdeg]) if outdeg else 1.,
     'g_breadth':np.log1p(max(Counter(dvalues).values(),default=0)),
     'g_repeated_author':1-nun/max(nc,1),'g_depth2_ratio':np.mean(np.asarray(dvalues)>=2) if dvalues else 0.,
     'e_log_duration':np.log1p(max(span,0)),
     'e_log_rate':np.log1p(len(times)/(1+max(tlist[-1],1))) if times else 0.,
     'e_first_delay':np.log1p(tlist[0]) if times else 0.,
     'e_median_delay':np.log1p(med),'e_p90_delay':np.log1p(p90),
     'e_frac_1h':float(early/max(len(times),1)),
     'e_frac_6h':float(np.sum(tlist<=360)/max(len(times),1)),
     'e_burst_early':float(early/(1+np.sqrt(max(len(times),1)))),
     'e_parent_lag':np.log1p(np.median([c[2]-p[2] for p,c in edges if c[2]>=p[2]])) if edges else 0.,
     'e_log_independent_branches':np.log1p(firstroot)
    }
    # Manually specified, label-blind fuzzy ontology rules. Their inputs are observed structural
    # and timing features; do NOT describe these as semantic stance or truth checking.
    g=f
    f.update({
      'o_rapid_independent_burst':float(expit(g['e_burst_early']-1.5)*expit(nun-4)*g['g_user_ratio']),
      'o_deep_persistence':float(expit(max(dvalues,default=0)-3)*expit(np.log1p(max(times,default=0))-3)),
      'o_broad_amplification':float(expit(root_degree-4)*expit(nun-7)),
      'o_delayed_revival':float(expit(np.log1p(p90)-5)*expit(1.5-g['e_frac_1h']*3)),
      'o_multi_branch_diffusion':float(expit(root_degree-3)*expit(max(dvalues,default=0)-2)*g['g_user_ratio']),
      'o_temporal_concentration':float(expit(2.5*g['e_frac_1h']-1)*expit(nc-6))
    })
    return f, {'nodes':nc,'edges':len(edges),'observed_times':len(times),'unique_users':nun}


def assemble_features(frame,base:Path,output:Path,windows):
    reports=[];feat=[];root=output/'audit'; root.mkdir(parents=True,exist_ok=True)
    for k,row in enumerate(frame.itertuples(index=False)):
        path=base/row.dataset/'tree'/f'{row.id}.txt'
        obj=parse_tree(path)
        tweetids=Counter(v[1] for v in obj['nodes'] if v[1]!='ROOT')
        repeated_tweet=sum(1 for v in tweetids.values() if v>1)
        reports.append({'dataset':row.dataset,'id':row.id,'node_total':len(obj['nodes']),'edge_total':len(obj['edges']),
                        'reused_tweet_ids_inside_tree':repeated_tweet,
                        'extra_nodes_sharing_tweet_id':sum(v-1 for v in tweetids.values() if v>1),
                        **{f'issue_{z}':v for z,v in obj['issues'].items()},'bad_lines_preview':' ; '.join(obj['examples'])})
        for w in windows:
            f,s=snapshot_features(obj,w)
            feat.append({'dataset':row.dataset,'id':row.id,'window':str(w),**f,**{f'visible_{z}':v for z,v in s.items()}})
        if (k+1)%500==0:print(f'parsed trees: {k+1}/{len(frame)}',flush=True)
    qa=pd.DataFrame(reports).fillna(0);qa.to_csv(root/'tree_integrity_by_source.csv',index=False)
    feats=pd.DataFrame(feat).fillna(0)
    if not np.isfinite(feats[G_COLS+R_COLS+O_COLS].to_numpy(dtype=float)).all():raise ValueError('Nonfinite features')
    feats.to_csv(output/'all_temporal_features.csv',index=False)
    return feats,qa



RULE_DESC={
 'o_rapid_independent_burst':('RapidIndependentBurst','High early activity AND distinct participating users'),
 'o_deep_persistence':('DeepPersistentDiscussion','Deep reply chain AND long observed timespan'),
 'o_broad_amplification':('BroadAmplification','High source degree AND many distinct users'),
 'o_delayed_revival':('DelayedRevival','Late response tail AND low early concentration'),
 'o_multi_branch_diffusion':('MultiBranchDiffusion','Several source branches AND deep cascades with distinct users'),
 'o_temporal_concentration':('TemporalConcentration','Most activity arrives early AND sufficient observations')
}

def emit_ontology_rdf(out:Path,features:pd.DataFrame,samples_per_dataset=12):
    """Materialize a sample of label-blind fuzzy rule scores as RDF provenance.
    This is NOT OWL-DL reasoning or veracity fact checking.
    """
    rules=[{'feature':k,'rule_name':v[0],'premise':v[1],'label_blind':True,'is_owl_dl_reasoning':False} for k,v in RULE_DESC.items()]
    pd.DataFrame(rules).to_csv(out/'ontology_rule_catalog.csv',index=False)
    ttl=['@prefix ex: <https://example.org/rumordetection/> .',
         '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
         '@prefix prov: <http://www.w3.org/ns/prov#> .',
         'ex:PropagationObservation a ex:EvidenceClass .',
         'ex:RuleActivation a ex:EvidenceClass .']
    rows=[]
    for dataset in ['twitter15','twitter16']:
        subset=features[(features.dataset==dataset)&(features.window=='full')].head(samples_per_dataset)
        for r in subset.itertuples(index=False):
            uid=f'{dataset}_{r.id}';ttl.append(f'ex:thread_{uid} a ex:PropagationObservation .')
            for name,(rule,premise) in RULE_DESC.items():
                value=float(getattr(r,name));i=f'{uid}_{name}'
                if value<=1e-6:continue
                ttl += [f'ex:act_{i} a ex:RuleActivation ; ex:rule ex:{rule} ; ex:confidence "{value:.6f}"^^xsd:decimal ; prov:wasDerivedFrom ex:thread_{uid} .']
                rows.append({'dataset':dataset,'id':r.id,'rule':rule,'score':value,'time_window':'full'})
    (out/'ontology_provenance_sample.ttl').write_text('\n'.join(ttl)+'\n',encoding='utf-8')
    pd.DataFrame(rows,columns=['dataset','id','rule','score','time_window']).to_csv(out/'ontology_provenance_sample.csv',index=False)

def build_text_encoder(train,valid,extra=None,dim=45):
    s_train=[str(x) for x in train]
    others=list(valid) if extra is None else list(valid)+list(extra)
    vw=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=16000,sublinear_tf=True,strip_accents='unicode')
    vc=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=22000,sublinear_tf=True)
    X=hstack([vw.fit_transform(s_train),vc.fit_transform(s_train)]).tocsr()
    Xt=hstack([vw.transform(others),vc.transform(others)]).tocsr()
    nk=min(dim,X.shape[0]-1,X.shape[1]-1)
    if nk<2:raise ValueError('Text input too small for SVD')
    svd=TruncatedSVD(n_components=nk,random_state=42,n_iter=4)
    tr=svd.fit_transform(X).astype(np.float32);te=svd.transform(Xt).astype(np.float32)
    return tr,te[:len(valid)],(te[len(valid):] if extra is not None else None)


def groupids(sub):
    # Source-only union-find of exact and near-duplicate texts, preventing CV overlap.
    # No target/test labels or texts enter this grouped-fold construction.
    z=sub.normalized.fillna('').astype(str).tolist();n=len(z)
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=root(i),root(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    seen={}
    for i,t in enumerate(z):
        if t in seen:union(i,seen[t])
        else:seen[t]=i
    vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,max_features=80000,sublinear_tf=True)
    mat=vec.fit_transform(z)
    for i in range(0,n,96):
        m=(mat[i:i+96]@mat.T).tocoo()
        for jj,k,sim in zip(m.row,m.col,m.data):
            j=i+jj
            if j>=k or sim<.93:continue
            if len(z[j])>=22 and len(z[k])>=22:union(j,k)
    return np.array([str(root(i)) for i in range(n)])


def make_predictor(seed:int):
    # Frozen architecture; parameter values are predeclared, not selected on target labels.
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=230,depth=5,learning_rate=0.055,l2_leaf_reg=7,loss_function='Logloss',verbose=False,random_seed=seed,thread_count=2,allow_writing_files=False)
    except ImportError:
        return HistGradientBoostingClassifier(max_iter=170,max_leaf_nodes=17,l2_regularization=6,learning_rate=.06,random_state=seed)


def features_for(df,cols):
    return np.nan_to_num(df[cols].to_numpy(dtype=np.float32),nan=0,posinf=0,neginf=0)


def train_and_predict(tr,va,xt_train,xt_valid,cols,seed,placebo=None):
    # Text scores learned with model A separately. For the matched ablation B/C/D/E,
    # ALL use same text projection and same CatBoost implementation/hyperparameters.
    xtr=np.hstack([xt_train,features_for(tr,cols)])
    xva=np.hstack([xt_valid,features_for(va,cols)])
    if placebo is not None:
        ix=cols.index(O_COLS[0]);rng=np.random.default_rng(placebo)
        for x,frame in [(xtr,tr),(xva,va)]:
            # Shuffle ontology rule *vectors* conditionally on log tree size; retrain for placebo.
            bins=np.floor(np.log2(1+np.expm1(frame.g_log_nodes.to_numpy(dtype=float)))).astype(int)
            for b in np.unique(bins):
                inds=np.where(bins==b)[0]
                if len(inds)>1:x[np.ix_(inds,np.arange(xt_train.shape[1]+ix,x.shape[1]))]=x[np.ix_(rng.permutation(inds),np.arange(xt_train.shape[1]+ix,x.shape[1]))]
    mdl=make_predictor(seed);mdl.fit(xtr,tr.label.to_numpy(dtype=int))
    return mdl.predict_proba(xva)[:,1]


def base_metric(y,p,t):
    pred=(np.asarray(p)>=t).astype(int);cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    return {'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)), 'accuracy':float(accuracy_score(y,pred)), 'rumor_recall':float(recall_score(y,pred,zero_division=0)), 'rumor_precision':float(precision_score(y,pred,zero_division=0)),'mcc':float(matthews_corrcoef(y,pred)), 'auprc':float(average_precision_score(y,p)) if len(set(y))>1 else float('nan'),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def select_threshold(y,p):
    thresholds=np.arange(.25,.751,.025)
    score=[f1_score(y,np.asarray(p)>=x,average='macro',zero_division=0) for x in thresholds]
    winners=np.flatnonzero(np.isclose(score,max(score),atol=1e-10))
    return float(thresholds[min(winners,key=lambda k:abs(thresholds[k]-.5))])


def get_fold_probs(source,features,window,seed=42,folds=3):
    cols={'A':[], 'B':G_COLS,'C':G_COLS+R_COLS,'D':G_COLS+R_COLS+O_COLS}
    dataset=source.merge(features[features.window==str(window)],on=['dataset','id'],validate='one_to_one')
    if len(dataset)!=len(source):raise RuntimeError('Join mismatch')
    groups=groupids(dataset); y=dataset.label.to_numpy(dtype=int)
    ccv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    out={k:np.zeros(len(dataset),dtype=float) for k in cols};seen=np.zeros(len(dataset),dtype=int)
    for fold,(it,iv) in enumerate(ccv.split(dataset,y,groups)):
        a=dataset.iloc[it];b=dataset.iloc[iv]
        xa,xb,_=build_text_encoder(a.source_text,b.source_text)
        for k,col in cols.items():
            out[k][iv]=train_and_predict(a,b,xa,xb,col,seed+fold)
        seen[iv]+=1
    if not (seen==1).all():raise RuntimeError('OOF incomplete')
    return dataset,out,{k:select_threshold(y,p) for k,p in out.items()}


def paired_bootstrap(y,p1,p0,t,seed=42,n_boot=1000):
    rng=np.random.default_rng(seed);y=np.asarray(y)
    if len(y)<5:return (float('nan'),float('nan'))
    vals=[]
    for i in range(n_boot):
        ix=rng.integers(0,len(y),len(y))
        vals.append(f1_score(y[ix],np.asarray(p1)[ix]>=t,average='macro',zero_division=0)-f1_score(y[ix],np.asarray(p0)[ix]>=t,average='macro',zero_division=0))
    return tuple(float(v) for v in np.quantile(vals,[.025,.975]))


def evaluate(frame,features,overlap,out,windows=TIME_WINDOWS,placebos=19,seed=42,bootstrap=1000,external=None):
    out.mkdir(exist_ok=True,parents=True)
    metric_rows=[];prediction_rows=[];placebo_rows=[];selection_rows=[]
    for w in windows:
      for origin,target in [('twitter15','twitter16'),('twitter16','twitter15')]:
        source=frame[frame.dataset==origin].copy().reset_index(drop=True)
        test=frame[(frame.dataset==target)&(~frame.id.isin(overlap[target]))].copy().reset_index(drop=True)
        if len(test)<(25 if max(EXPECTED.values())<100 else 100):raise RuntimeError(f'Too few clean targets: {origin}->{target}: {len(test)}')
        sa,oo,thr=get_fold_probs(source,features,w,seed=seed)
        # Frozen source-only selection with small predetermined complexity penalty.
        ss={k:base_metric(sa.label.to_numpy(),p,thr[k])['macro_f1']-(.002 if k=='D' else 0) for k,p in oo.items()}
        selected=max(sorted(ss),key=lambda k:ss[k])
        selection_rows.append({'direction':f'{origin}_to_{target}','window':w,'selected_by_source_oof':selected,**{f'thr_{k}':v for k,v in thr.items()},**{f'oof_{k}':v for k,v in ss.items()}})
        train=source.merge(features[features.window==str(w)],on=['dataset','id'],validate='one_to_one')
        hold=test.merge(features[features.window==str(w)],on=['dataset','id'],validate='one_to_one')
        train_text,test_text,_=build_text_encoder(train.source_text,hold.source_text)
        probs={}
        for k,col in {'A':[], 'B':G_COLS,'C':G_COLS+R_COLS,'D':G_COLS+R_COLS+O_COLS}.items():
            probs[k]=train_and_predict(train,hold,train_text,test_text,col,seed)
            metr=base_metric(hold.label.to_numpy(dtype=int),probs[k],thr[k]); metric_rows.append({'window':w,'direction':f'{origin}_to_{target}','config':k,'n_train':len(train),'n_test':len(hold),**metr})
        # Operational G model is selected solely from source-domain grouped OOF.
        probs['G']=probs[selected].copy();thr['G']=thr[selected]
        metric_rows.append({'window':w,'direction':f'{origin}_to_{target}','config':'G_source_selected','n_train':len(train),'n_test':len(hold),'selected_base':selected,
                          **base_metric(hold.label.to_numpy(dtype=int),probs['G'],thr['G'])})
        y=hold.label.to_numpy(dtype=int)
        for k in probs:
            for r,p in zip(hold.itertuples(index=False),probs[k]):
                prediction_rows.append({'direction':f'{origin}_to_{target}','window':w,'config':k,'dataset':r.dataset,'id':r.id,'label':r.label,'probability':float(p),'threshold_source_oof':thr[k]})
        # Compare D to C using the SAME source-only threshold (C's). This isolates features.
        t=thr['C'];d=base_metric(y,probs['D'],t)['macro_f1']-base_metric(y,probs['C'],t)['macro_f1']
        lo,hi=paired_bootstrap(y,probs['D'],probs['C'],t,seed,bootstrap)
        metric_rows.append({'window':w,'direction':f'{origin}_to_{target}','config':'D-C_shared_threshold', 'n_train':len(train),'n_test':len(hold),'macro_f1':d,'ci_lower':lo,'ci_upper':hi})
        if w=='full':
            for rep in range(placebos):
                # Placebo labels never used to shuffle; retrain with shuffled rule vectors.
                pp=train_and_predict(train,hold,train_text,test_text,G_COLS+R_COLS+O_COLS,seed,placebo=seed+10000+rep)
                placebo_rows.append({'direction':f'{origin}_to_{target}','window':w,'repeat':rep,'macro_f1_shared_threshold':base_metric(y,pp,t)['macro_f1'],'real_ontology_macro_f1_shared_threshold':base_metric(y,probs['D'],t)['macro_f1']})
        print('done',origin,'->',target,'window=',w,'n=',len(hold),'selected=',selected,'D-C=',round(d,4),flush=True)
        write_outputs(out,metric_rows,prediction_rows,selection_rows,placebo_rows,partial=True)
    write_outputs(out,metric_rows,prediction_rows,selection_rows,placebo_rows,partial=False)
    placebo_tests=[]
    if placebo_rows:
      pdf=pd.DataFrame(placebo_rows)
      for (direction,window),pgrp in pdf.groupby(['direction','window']):
        real=pgrp.real_ontology_macro_f1_shared_threshold.iloc[0]
        null=pgrp.macro_f1_shared_threshold.to_numpy(dtype=float)
        placebo_tests.append({'direction':direction,'window':str(window),'real_macro_f1':real,
          'placebo_mean':float(null.mean()),'placebo_sd':float(null.std(ddof=1)) if len(null)>1 else 0.,
          'rank_p_upper':float((1+sum(null>=real))/(1+len(null))),
          'repeats':len(null),'note':'one-sided Monte Carlo rank p, exploratory conditional placebo'})
    pd.DataFrame(placebo_tests).to_csv(out/'placebo_rank_summary.csv',index=False)
    for p in placebo_tests:
      metric_rows.append({'window':p['window'],'direction':p['direction'],'config':'E_placebo_mean',
                          'macro_f1':p['placebo_mean'],'placebo_sd':p['placebo_sd']})
    pd.DataFrame(metric_rows).to_csv(out/'metrics.csv',index=False)
    return metric_rows


def write_outputs(out,metrics,preds,select,placebos,partial=False):
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(select).to_csv(out/'source_only_selection.csv',index=False)
    pd.DataFrame(placebos).to_csv(out/'placebo_retrained.csv',index=False)
    (out/'run_state.json').write_text(json.dumps({'partial':partial,'metrics_rows':len(metrics),'prediction_rows':len(preds)},indent=2))


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'FROZEN_CONFIG.json').write_text(json.dumps({'version':'2026-11-05-final1',
         'seed':args.seed,'time_windows':args.windows,'placebo_repeats':args.placebo_repeats,'bootstrap':args.bootstrap,
         'expected_dataset_sizes':EXPECTED,'original_git_sha':ORIGINAL_SHA,'source_text_git_sha':SOURCE_SHA,
         'test_driven_tuning_allowed':False,'task':'binary rumor vs nonrumor','ontology':'structural fuzzy rules, not OWL-DL',
         'source_pipeline_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    base=prepare_data(Path(args.workdir),Path(args.input_root) if args.input_root else None,not args.no_download)
    data,inventory=load_datasets(base,strict=True)
    qa_dir=out/'audit';qa_dir.mkdir(exist_ok=True)
    inventory.to_csv(qa_dir/'inventory.csv',index=False)
    data.drop(columns=['source_text']).to_csv(qa_dir/'source_and_label_manifest.csv',index=False)
    summary,blocked=duplicate_audit(data,qa_dir)
    feats,tqa=assemble_features(data,base,out,args.windows)
    emit_ontology_rdf(out,feats)
    data.to_csv(out/'canonical_labeled_sources.csv',index=False)
    if args.audit_only:
        print('Audit only complete:',summary);return
    if args.external_dir:
        (out/'INDEPENDENT_EXTERNAL_STATUS.txt').write_text('External dataset path supplied, but additional dataset is NOT automatically equated with rumor/non-rumor. This pipeline only scores Twitter15/16. For a third dataset, label definitions, preprocessing, provenance, and overlap must first be independently validated. No third-dataset result is claimed.\n')
    else:
        (out/'INDEPENDENT_EXTERNAL_STATUS.txt').write_text('PENDING: no truly independent third dataset supplied. Twitter15/Twitter16 were used during model development; this is a cross-dataset robustness assessment, not untouched external confirmation.\n')
    evaluate(data,feats,blocked,out,args.windows,args.placebo_repeats,args.seed,args.bootstrap)
    with zipfile.ZipFile(out.parent/(out.name+'.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(out.parent))
    print('ZIP ready:',out.parent/(out.name+'.zip'))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--workdir',default='/kaggle/working/twitter_full_source')
    ap.add_argument('--input-root',default='/kaggle/input')
    ap.add_argument('--output',default='/kaggle/working/Twitter1516_Frozen_Full_Audit_Results')
    ap.add_argument('--windows',nargs='+',default=['60','360','1440','full'])
    ap.add_argument('--placebo-repeats',type=int,default=19)
    ap.add_argument('--bootstrap',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--no-download',action='store_true')
    ap.add_argument('--external-dir',default=None)
    main(ap.parse_args())
