#!/usr/bin/env python3
"""Analyze 216 independent ordinal blind ratings. No external packages required.
Usage: python analyze_annotations.py r1.csv r2.csv [out_folder]
"""
import csv,collections,json,math,sys,pathlib
FIELDS=['premise_supported','rule_semantically_valid','temporally_valid','explanation_usefulness']
VALID={'0','1','2'}
def load(path):
    with open(path,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Duplicate case_id in '+str(path))
    codes={r['annotator_code'] for r in rows};
    if len(codes)!=1:raise ValueError('A file must contain one reviewer code: '+str(codes))
    return codes.pop(),{r['case_id']:r for r in rows}
def kappa(a,b,weighted=False):
    n=len(a)
    if n==0:return None
    table=[[0]*3 for _ in range(3)]
    for x,y in zip(a,b):table[int(x)][int(y)]+=1
    left=[sum(r) for r in table];right=[sum(table[i][j] for i in range(3)) for j in range(3)]
    def d(i,j):return (i-j)**2/4 if weighted else (0 if i==j else 1)
    observed=sum(table[i][j]*d(i,j) for i in range(3) for j in range(3))/n
    expected=sum(left[i]*right[j]*d(i,j) for i in range(3) for j in range(3))/n**2
    return (1-observed/expected) if expected>0 else (1.0 if observed==0 else None)
def main(argv):
    if len(argv)<3:raise SystemExit('Usage: python analyze_annotations.py R1.csv R2.csv [outdir]')
    c1,a=load(argv[1]);c2,b=load(argv[2]);
    if c1==c2:raise ValueError('Both files have same rater ID')
    ids=sorted(set(a)&set(b));out=pathlib.Path(argv[3] if len(argv)>3 else '.');out.mkdir(parents=True,exist_ok=True)
    summary=[];dis=[]
    for field in FIELDS:
        both=[(cid,a[cid][field],b[cid][field]) for cid in ids if a[cid].get(field) in VALID and b[cid].get(field) in VALID]
        xa=[x[1] for x in both];xb=[x[2] for x in both];n=len(both)
        row=dict(criterion=field,n_both_scored=n,agreement=round(sum(x==y for x,y in zip(xa,xb))/n,4) if n else None,kappa_unweighted=kappa(xa,xb),kappa_quadratic_weighted=kappa(xa,xb,True),rater1_mean=round(sum(map(int,xa))/n,4) if n else None,rater2_mean=round(sum(map(int,xb))/n,4) if n else None,missing_or_NA=len(ids)-n)
        summary.append(row)
        for cid,x,y in both:
            if x!=y:dis.append(dict(case_id=cid,criterion=field,rater1=c1,score1=x,rater2=c2,score2=y))
    with (out/'interrater_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with (out/'disagreements.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=['case_id','criterion','rater1','score1','rater2','score2']);w.writeheader();w.writerows(dis)
    report={'raters':[c1,c2],'total_case_ids_in_both_files':len(ids),'fully_scored_both_raters':sum(all(a[cid].get(k) in VALID and b[cid].get(k) in VALID for k in FIELDS) for cid in ids),'scales':summary,'disagreement_rows':len(dis),'note':'Cohen kappa is not expert certification. NA is excluded pairwise; no inferred scores.'}
    (out/'agreement_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main(sys.argv)
