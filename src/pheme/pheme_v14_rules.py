"""v14 evidence-reliability ontology: thread-local, label-blind operational rules.

A rule activation means a recorded *cue*, never a verified true/false claim.
No global target statistics, PHEME event IDs or train/test labels are consumed.
Each record has source/reply provenance and uncertainty/reliability metadata.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

RULES14 = [
    'o14_direct_challenge_smoothed',
    'o14_independent_challenge_smoothed',
    'o14_author_evidence_disagreement',
    'o14_cited_denial_confidence',
    'o14_query_evidence_gap',
    'o14_direct_query_confidence',
    'o14_early_late_challenge_shift',
    'o14_parent_chain_challenge',
    'o14_different_author_challenge',
    'o14_source_uncertainty_interaction',
    'o14_semantic_challenge_tension',
    'o14_reply_sampling_reliability',
    'o14_author_diversity_reliability',
    'o14_citation_independence',
    'o14_corroboration_gap',
    'o14_dispute_depth_interaction',
    'o14_independent_challenge_entropy',
    'o14_lexical_evidence_balance',
]
DESCRIPTIONS14 = [
    'Smoothed observed challenge cues in source-directed replies',
    'Smoothed fraction of distinct authors presenting challenge cues',
    'Co-occurrence of supporting and challenging authors',
    'Denial cues co-occurring with a citation cue (not fact-checking)',
    'Evidence-request cues without cited supporting replies',
    'Smoothed source-directed verification requests',
    'Early-to-later change in challenge-cue rate',
    'Challenges replying to another sampled reply in the thread',
    'Challenges from authors different from the source author',
    'Source epistemic uncertainty and challenge requests',
    'Observed challenge cues weighted by semantic novelty (not contradiction)',
    'Fraction of thread replies observed in the predefined sampling window',
    'Effective diverse-author sample size (confidence proxy only)',
    'Citation cues supported by multiple independent authors',
    'Lack of citation-bearing support among sampled replies',
    'Conversation reply depth interacting with challenge evidence',
    'Entropy across authors with challenge and non-challenge replies',
    'Signed balance of support cues minus challenge cues',
]

def build_v14(df: pd.DataFrame, replies: pd.DataFrame):
    """Returns features, provenance audit. No labels/events may affect output."""
    grouped = {str(k):g for k,g in replies.groupby('thread_id', sort=False)}
    rows, audits = [], []
    for r in df.itertuples(index=False):
        thread = str(r.thread_id)
        grp = grouped.get(thread)
        out = {k:0. for k in RULES14}
        if grp is None or grp.empty:
            rows.append(out)
            continue
        grp = grp.sort_values(['rank'],kind='stable')
        n = len(grp)
        ids = grp.post_id.fillna('').astype(str).to_numpy()
        parents = grp.reply_to.fillna('').astype(str).to_numpy()
        users = grp.user_id.fillna('').astype(str).to_numpy()
        direct = grp.direct.to_numpy(bool)
        den = grp.denial.to_numpy(float)
        que = grp['query'].to_numpy(float)
        sup = grp.support.to_numpy(float)
        cit = grp.citation.to_numpy(float)
        un = grp.uncertainty.to_numpy(float)
        ch = np.maximum(den,que)
        src_user=str(r.source_user)
        valid = (users != '')
        authors = set(users[valid]); n_author=len(authors)
        chal_a=set(users[(ch>0)&valid]); support_a=set(users[(sup>0)&valid]); cit_a=set(users[(cit>0)&valid])
        # Evidence-rate shrinkage toward zero, with two pseudo-observations. No corpus labels are used.
        smooth=lambda hits,count: float(hits/(2+count))
        avg=lambda v: float(np.mean(v)) if len(v) else 0.
        out['o14_direct_challenge_smoothed']=smooth(ch[direct].sum(),int(direct.sum())) if direct.any() else 0.
        out['o14_independent_challenge_smoothed']=smooth(len(chal_a),n_author) if n_author else 0.
        out['o14_author_evidence_disagreement']=float(bool(chal_a and support_a and any(a!=b for a in chal_a for b in support_a)))
        out['o14_cited_denial_confidence']=smooth(np.sum(den*cit),n)*np.sqrt(min(n,8)/8)
        out['o14_query_evidence_gap']=avg(que)*(1-avg(cit*sup))
        out['o14_direct_query_confidence']=smooth(que[direct].sum(),int(direct.sum())) if direct.any() else 0.
        k=min(3,n)
        out['o14_early_late_challenge_shift']=avg(ch[k:])-avg(ch[:k]) if n>k else 0.
        known=set(ids)
        chain=np.array([(par in known and par!=tid) for par,tid in zip(parents,ids)],dtype=float)
        out['o14_parent_chain_challenge']=avg(ch*chain)
        out['o14_different_author_challenge']=avg(ch*(users!=src_user)*valid)
        source_unc=float(getattr(r,'source_uncertainty',0))
        out['o14_source_uncertainty_interaction']=source_unc*avg(ch+que)/2
        novelty=float(getattr(r,'e_semantic_novelty_rate',0))
        out['o14_semantic_challenge_tension']=avg(ch)*max(0.,min(1.,novelty))
        total=max(1.,float(getattr(r,'num_nodes',n+1))-1)
        out['o14_reply_sampling_reliability']=min(1.,n/total)
        out['o14_author_diversity_reliability']=len(authors)/max(1.,n)*np.sqrt(min(n,8)/8)
        out['o14_citation_independence']=avg(cit)*min(1.,len(cit_a)/max(1,np.sqrt(n_author)))
        out['o14_corroboration_gap']=avg(ch)*(1-avg(cit*sup))
        depth=float(getattr(r,'thread_depth',0))
        out['o14_dispute_depth_interaction']=avg(ch)*min(1.,np.log1p(max(depth,0))/np.log1p(20))
        # Bounded binary entropy of distinct authors who used any challenge cues.
        q=len(chal_a)/max(1,n_author)
        out['o14_independent_challenge_entropy']=float(-(q*np.log2(q)+(1-q)*np.log2(1-q))) if 0<q<1 else 0.
        out['o14_lexical_evidence_balance']=avg(sup)-avg(ch)
        if len(audits)<2000:
            # High-precision provenance for explainability: only when at least one explicit cue exists.
            observed=np.where((ch+cit+sup)>0)[0]
            if len(observed):
                audits.append(dict(thread_id=thread,n_sampled=n,n_authors=n_author,
                    source_id=thread,reply_evidence_json=json.dumps([
                      dict(id=str(ids[j]),parent=str(parents[j]),author=str(users[j]),
                           direct=bool(direct[j]),denial=bool(den[j]),query=bool(que[j]),
                           support=bool(sup[j]),citation_cue=bool(cit[j]))
                      for j in observed[:8]]),
                    rules_json=json.dumps({k:round(float(v),5) for k,v in out.items() if abs(v)>1.e-8})))
        rows.append(out)
    result = pd.DataFrame(rows,index=df.index)[RULES14].astype('float32')
    if result.shape[0]!=len(df) or not np.isfinite(result.to_numpy()).all():
        raise AssertionError('invalid ontology v14 features')
    return result,pd.DataFrame(audits)


def assert_label_blind(df,replies):
    a,_=build_v14(df,replies)
    other=df.copy()
    if 'label' in other:other['label']=1-other['label'].astype(int)
    if 'event_id' in other:other['event_id']='perturbed'
    b,_=build_v14(other,replies)
    np.testing.assert_array_equal(a.to_numpy(),b.to_numpy())
    return True
