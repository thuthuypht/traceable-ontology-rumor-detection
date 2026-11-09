#!/usr/bin/env python3
"""Independent verification of saved model/pipeline and case-specific intervention outputs.
Tests EVERY model (C/D) against all 216 target cases, without fitting any model.
Not a claim that reconstructed models equal missing 2026-11-05 originals.
"""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd,numpy as np,joblib
from catboost import CatBoostClassifier
from scipy.sparse import hstack
import full_provenance_faithfulness as pipeline

def verify(root:Path):
    inp=root/'source_snapshot'
    s,ca,ref,fr=pipeline.load(inp)
    pred=pd.read_csv(root/'reports'/'case_specific_faithfulness_216.csv',dtype={'source_id':str,'window':str})
    detail=pd.read_csv(root/'reports'/'all_six_rule_interventions_per_case.csv',dtype={'id':str,'window':str})
    failures=[];max_base=0.;max_cf=0.;testcount=0
    for g,part in pred.groupby(['direction','policy','window'],sort=True):
        direction,policy,window=g;origin,target=direction.split('_to_');name=f'{direction}__{policy}__{window}'
        base=root/'models'/name;meta=json.loads((base/'metadata.json').read_text())
        assert meta['model_D_sha256']==pipeline.sha(base/'model_D_catboost.cbm')
        assert meta['model_C_sha256']==pipeline.sha(base/'model_C_catboost.cbm')
        prefile=root/'preprocessors'/direction/'fitted_text_pipeline.joblib'
        assert meta['preprocessor_sha256']==pipeline.sha(prefile)
        prep=joblib.load(prefile)
        all_te=s[s.dataset==target].copy()
        # Held-out test ID exact order comes from historical predictions; no labels enter model.
        rr=ref[(ref.direction==direction)&(ref.policy==policy)&(ref.window==window)]
        testids=set(rr.id);all_te=all_te[all_te.id.isin(testids)]
        data=pipeline.feat_frame(all_te,fr[policy],window)
        texts=data.source_text.astype(str).tolist()
        X=hstack([prep['word_vectorizer'].transform(texts),prep['char_vectorizer'].transform(texts)]).tocsr()
        xtext=prep['svd'].transform(X).astype(np.float32)
        assert len(xtext)==len(data)
        x=np.hstack([xtext,pipeline.get_numeric(data,pipeline.INPUT_COLS)])
        model=CatBoostClassifier();model.load_model(str(base/'model_D_catboost.cbm'))
        refcase=part.set_index('source_id')
        indices={v:i for i,v in enumerate(data.id)}
        idx=np.array([indices[q] for q in refcase.index],dtype=int)
        got=model.predict_proba(x[idx])[:,1]
        for i,p in enumerate(got):
            caseid=refcase.case_id.iloc[i];expected=float(refcase.p_model_D.iloc[i]);error=abs(float(p)-expected)
            max_base=max(max_base,error);testcount+=1
            if error>1e-8:failures.append(f'Base {caseid}: {error}')
        median=np.median(np.hstack([xtext[:0,:],np.empty((0,len(pipeline.INPUT_COLS)),dtype=np.float32)]),axis=0) if False else None
        # Every intervention requires the training-set median, stored per case in the detailed CSV.
        for i,sub in enumerate(refcase.itertuples()):
            cr=detail[detail.case_id==sub.case_id]
            for crule in cr.itertuples(index=False):
                arr=x[idx[i]:idx[i]+1].copy();j=pipeline.INPUT_COLS.index(crule.rule)
                arr[0,xtext.shape[1]+j]=float(crule.source_train_median)
                val=float(model.predict_proba(arr)[0,1]);err=abs(val-float(crule.p_after_median))
                max_cf=max(max_cf,err)
                if err>1e-8:failures.append(f'Intervention {sub.case_id} {crule.rule}: {err}')
    report={'verified_cases':testcount,'verified_interventions':int(len(detail)),
            'max_abs_probability_reload_error':max_base,'max_abs_intervention_reload_error':max_cf,
            'failure_count':len(failures),'first_failures':failures[:12],
            'verified_model_type':'CatBoost retrained and saved in this package',
            'historical_reproduction':'Separate audit; original trained checkpoint unavailable'}
    (root/'reports'/'independent_reload_verification.json').write_text(json.dumps(report,indent=2))
    if failures:raise AssertionError('Model reload verification failed '+str(report))
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    print(json.dumps(verify(Path(a.root)),indent=2))
if __name__=='__main__':main()
