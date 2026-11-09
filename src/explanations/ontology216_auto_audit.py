#!/usr/bin/env python3
"""Frozen automatic audit of 216 structural-fuzzy-ontology explanations.

This audit does NOT produce human expert judgments, OWL entailments, or truth labels.
It checks reproducibility of predefined scores from visible evidence, optionally
from the original Twitter15/16 propagation trees, sampled provenance, and
aggregated model sensitivity. No labels are used to set thresholds or tune models.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, io, json, math, os, re, shutil, statistics, sys, urllib.request, zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

ORIGINAL_SHA = '6a33ebd91d4cafeb5a42eabc5727226fec7bb0e5'
ORIGINAL_URL = f'https://codeload.github.com/gszswork/Twitter15_16_dataset/zip/{ORIGINAL_SHA}'
RULES = ('o_rapid_independent_burst','o_deep_persistence','o_broad_amplification',
         'o_delayed_revival','o_multi_branch_diffusion','o_temporal_concentration')
RULE_NAMES = {'RapidIndependentBurst':RULES[0], 'DeepPersistentDiscussion':RULES[1],
 'BroadAmplification':RULES[2], 'DelayedRevival':RULES[3],
 'MultiBranchDiffusion':RULES[4], 'TemporalConcentration':RULES[5]}
TOL_SCORE = 0.005  # Feature values visible to reviewers are rounded to 4 decimals.
TOL_FEATURE = 0.005

def sigmoid(x):
    return 1.0/(1.0+math.exp(-max(-80.0,min(80.0,float(x)))))

def readcsv(path):
    with open(path,encoding='utf-8-sig',newline='') as fp: return list(csv.DictReader(fp))

def writecsv(path,rows,fieldnames):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=fieldnames,extrasaction='ignore');w.writeheader();w.writerows(rows)

def parse_evidence(text):
    result={}
    for part in (text or '').split(';'):
        m=re.fullmatch(r'\s*([a-z][a-z0-9_]*)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*',part)
        if m: result[m.group(1)]=float(m.group(2))
    return result

def fuzzy_scores(e):
    # Return None when one of the required inputs is missing.
    def v(x):return e[x]
    out={}
    def calculate(name,keys,formula):
        if all(k in e and math.isfinite(e[k]) for k in keys):
            try: out[name]=float(formula())
            except (OverflowError,ValueError,ZeroDivisionError):out[name]=None
        else:out[name]=None
    calculate('o_rapid_independent_burst',('e_burst_early','g_log_users','g_user_ratio'),
              lambda: sigmoid(v('e_burst_early')-1.5)*sigmoid(math.expm1(v('g_log_users'))-4)*v('g_user_ratio'))
    calculate('o_deep_persistence',('g_max_depth','e_log_max_time'),
              lambda: sigmoid(math.expm1(v('g_max_depth'))-3)*sigmoid(v('e_log_max_time')-3))
    calculate('o_broad_amplification',('g_root_degree','g_log_users'),
              lambda: sigmoid(math.expm1(v('g_root_degree'))-4)*sigmoid(math.expm1(v('g_log_users'))-7))
    calculate('o_delayed_revival',('e_p90_delay','e_frac_1h'),
              lambda: sigmoid(v('e_p90_delay')-5)*sigmoid(1.5-v('e_frac_1h')*3))
    calculate('o_multi_branch_diffusion',('g_root_degree','g_max_depth','g_user_ratio'),
              lambda: sigmoid(math.expm1(v('g_root_degree'))-3)*sigmoid(math.expm1(v('g_max_depth'))-2)*v('g_user_ratio'))
    calculate('o_temporal_concentration',('e_frac_1h','visible_nodes'),
              lambda: sigmoid(2.5*v('e_frac_1h')-1)*sigmoid(v('visible_nodes')-6))
    return out

def parse_node(txt):
    x=ast.literal_eval(txt.strip())
    if not isinstance(x,list) or len(x)!=3:raise ValueError('Node must contain uid, tweet_id, delay')
    a,b,c=x
    delay=float(c)
    if not math.isfinite(delay):raise ValueError('Nonfinite time')
    return str(a),str(b),delay

def parse_tree(src):
    nodes=set();edges=set();bad=Counter()
    for line in src.splitlines():
        line=line.strip()
        if not line:continue
        try:
            left,right=line.split('->',1)
            p,c=parse_node(left),parse_node(right)
            if p==c:bad['self_edges']+=1;continue
            if (p,c) in edges:bad['duplicate_edges']+=1
            edges.add((p,c));nodes.update((p,c))
            if min(p[2],c[2])<0:bad['negative_delay']+=1
            if c[2]<p[2]-1e-8:bad['backward_edges']+=1
        except (SyntaxError,ValueError,TypeError):bad['malformed_lines']+=1
    return {'nodes':nodes,'edges':edges,'issues':dict(bad)}

def strict_tree(old):
    valid=sorted([(p,c) for p,c in old['edges'] if p[2]>=0 and c[2]>=0 and c[2]+1e-9>=p[2] and p!=c],
       key=lambda a:(a[0][2],a[1][2],a[0][0],a[1][0],a[1][1]))
    children=defaultdict(list)
    for p,c in valid:children[p].append(c)
    roots=[n for n in old['nodes'] if n[1]=='ROOT'] or [('ROOT','ROOT',0.0)]
    visited=set(roots);q=deque(sorted(roots));kept=set()
    while q:
        p=q.popleft()
        for c in sorted(children.get(p,[]),key=lambda x:(x[2],x[0],x[1])):
            if c not in visited:visited.add(c);kept.add((p,c));q.append(c)
    return {'nodes':visited,'edges':kept,'issues':old['issues']}

def quantile90(xs):
    if not xs:return 0.0
    ys=sorted(xs);pos=0.9*(len(ys)-1);lo=int(pos);hi=min(lo+1,len(ys)-1)
    return ys[lo]*(1-(pos-lo))+ys[hi]*(pos-lo)

def snapshot(tree,window):
    bound=float(window) if str(window).lower()!='full' else None
    ns={x for x in tree['nodes'] if x[2]>=0 and (bound is None or x[2]<=bound)}
    es=[(p,c) for p,c in tree['edges'] if p in ns and c in ns and c[2]>=p[2]]
    real=[v for v in ns if v[1]!='ROOT']
    parent={c:p for p,c in es};child=defaultdict(list)
    for p,c in es:child[p].append(c)
    starts=[v for v in real if v not in parent or parent[v][1]=='ROOT']
    depths={v:0 for v in starts};q=deque(starts)
    while q:
        v=q.popleft()
        for c in child[v]:
            d=depths[v]+1
            if d>len(ns):continue
            if c not in depths or depths[c]>d:depths[c]=d;q.append(c)
    times=sorted(v[2] for v in real if v[2]>0)
    dep=max([depths.get(v,0) for v in real],default=0)
    nc=len(real);nun=len({v[0] for v in real})
    root_degree=sum(len(child[v]) for v in starts)
    early=sum(t<=60 for t in times)
    p90=quantile90(times)
    e={'e_burst_early':early/(1+math.sqrt(max(len(times),1))),
       'g_log_users':math.log1p(nun),'g_user_ratio':nun/max(nc,1),
       'g_root_degree':math.log1p(root_degree),'g_max_depth':math.log1p(dep),
       'e_log_max_time':math.log1p(max(times,default=0)),
       'e_p90_delay':math.log1p(p90),'e_frac_1h':early/max(len(times),1),
       'visible_nodes':nc, 'visible_users':nun,
       'visible_edges':len(es), 'raw_nodes':len(tree['nodes']),
       'raw_edges':len(tree['edges'])}
    return e

def get_trees(cases, raw_zip, work, auto_download=False):
    requested={(x['dataset'],x['source_id']) for x in cases}
    base=work/'raw_trees';base.mkdir(parents=True,exist_ok=True)
    want={f'{ds}/tree/{tw}.txt':(ds,tw) for ds,tw in requested}
    sources={}
    for rel,(ds,tw) in want.items():
        p=base/rel
        if p.exists():sources[(ds,tw)]=p.read_text(encoding='utf-8',errors='replace')
    missing=set(requested)-set(sources)
    if not missing:return sources,'LOCAL_CACHED'
    source_zip=Path(raw_zip) if raw_zip else None
    if source_zip and not source_zip.exists():source_zip=None
    if source_zip is None and auto_download:
        source_zip=work/'twitter1516_pinned_full.zip'
        if not source_zip.exists():
            try:
                print('Downloading pinned original Twitter15/16 trees (~ZIP), Kaggle Internet must be ON...',flush=True)
                with urllib.request.urlopen(ORIGINAL_URL,timeout=65) as resp, open(source_zip,'wb') as dst:
                    shutil.copyfileobj(resp,dst)
                print('Downloaded',source_zip.stat().st_size,'bytes',flush=True)
            except Exception as exc:
                print('SOURCE_DOWNLOAD_WARNING:',type(exc).__name__,str(exc)[:250],flush=True)
                source_zip.unlink(missing_ok=True);source_zip=None
    if source_zip:
        try:
            with zipfile.ZipFile(source_zip) as z:
                for inf in z.infolist():
                    name=inf.filename
                    for rel,(ds,tw) in want.items():
                        if (ds,tw) not in missing:continue
                        if name.endswith('/'+rel) or name==rel:
                            body=z.read(inf).decode('utf-8-sig',errors='replace')
                            sources[(ds,tw)]=body
                            p=base/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body,encoding='utf-8')
                print('Matched original trees:',len(sources),'of',len(requested),flush=True)
        except Exception as exc:print('TREE_ZIP_READ_WARNING',type(exc).__name__,str(exc)[:200],flush=True)
    return sources, 'PINNED_GITHUB_ZIP' if source_zip else 'VISIBLE_EVIDENCE_ONLY'

def summarize(rows):
    n=len(rows);cs=Counter(x['auto_status'] for x in rows)
    stats={'total_cases':n,'overall_status_counts':dict(cs),
       'full_raw_trees_verified':sum(x['raw_status']=='PASS' for x in rows),
       'visible_score_verified':sum(x['visible_status']=='PASS' for x in rows),
       'raw_score_failures':sum(x['raw_status']=='FAIL' for x in rows),
       'case_specific_model_faithfulness_measured':False,
       'human_expert_validation_completed':False,'claim_truth_verification_completed':False}
    rules=[]
    for rule in RULES:
        grp=[x for x in rows if x['rule_feature']==rule]
        rules.append({'rule':rule,'n':len(grp),'raw_pass':sum(x['raw_status']=='PASS' for x in grp),
           'visible_pass':sum(x['visible_status']=='PASS' for x in grp),
           'insufficient_without_raw':sum(x['visible_status']=='INSUFFICIENT' for x in grp),
           'failed':sum(x['auto_status']=='FAIL' for x in grp),
           'score_saturated_ge_095':sum(float(x['stored_rule_score'])>=.95 for x in grp),
           'provenance_sample_matches':sum(x['sample_provenance_status']=='PASS' for x in grp),
           'provenance_not_present':sum(x['sample_provenance_status']=='NOT_SAMPLED' for x in grp)})
    return stats,rules

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',required=True)
    ap.add_argument('--key',required=True)
    ap.add_argument('--provenance',default='')
    ap.add_argument('--rule-sensitivity',default='')
    ap.add_argument('--raw-zip',default='')
    ap.add_argument('--auto-download',action='store_true')
    ap.add_argument('--output-dir',default='/kaggle/working/ontology216_audit')
    args=ap.parse_args()
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    cases=readcsv(args.cases);key=readcsv(args.key)
    if len(cases)!=216 or len(key)!=216:raise RuntimeError('Expected exactly 216 case records and researcher keys')
    if len({x['case_id'] for x in cases})!=216:raise RuntimeError('Duplicate case_id')
    keys={x['case_id']:x for x in key}
    if set(keys)!={x['case_id'] for x in cases}:raise RuntimeError('Researcher key is not aligned')
    provenance=readcsv(args.provenance) if args.provenance and Path(args.provenance).exists() else []
    prov={(x['dataset'],x['id'],x['rule'],x['time_window']):x for x in provenance}
    sensitivity=readcsv(args.rule_sensitivity) if args.rule_sensitivity and Path(args.rule_sensitivity).exists() else []
    sens={(x['direction'],x['rule']):x for x in sensitivity}
    raw,raw_source=get_trees(cases,args.raw_zip,root,args.auto_download)
    out=[];raw_cache={}
    for r in cases:
        keyid=(r['dataset'],r['source_id']); rule=r['rule_feature']; score=float(r['model_rule_score'])
        if rule not in RULES:raise RuntimeError('Unexpected rule: '+rule)
        k=keys[r['case_id']]
        if k['dataset']!=r['dataset'] or k['source_id']!=r['source_id'] or k['rule']!=rule:
            raise RuntimeError('Researcher case mapping disagreement: '+r['case_id'])
        observed=parse_evidence(r['observed_structural_evidence'])
        visible=fuzzy_scores(observed)[rule]
        vstatus='INSUFFICIENT' if visible is None else 'PASS' if abs(visible-score)<=TOL_SCORE else 'FAIL'
        actual=None;feat={};rstatus='NOT_AVAILABLE';diffs={};issue={};reason=[]
        if keyid in raw:
            try:
                if keyid not in raw_cache:raw_cache[keyid]=parse_tree(raw[keyid])
                tree=raw_cache[keyid]
                if r['tree_policy']=='strict_reachable_tree':tree=strict_tree(tree)
                elif r['tree_policy']!='original':raise ValueError('Unknown policy')
                feat=snapshot(tree,r['time_window'])
                actual=fuzzy_scores(feat)[rule]
                if actual is None:raise RuntimeError('Could not calculate score on raw tree')
                rstatus='PASS' if abs(actual-score)<=TOL_SCORE else 'FAIL'
                diffs={name:abs(float(value)-float(feat[name])) for name,value in observed.items() if name in feat}
                if diffs and max(diffs.values())>TOL_FEATURE:
                    rstatus='FAIL';reason.append('Evidence values do not match frozen graph extraction')
                issue=tree['issues']
            except Exception as exc:
                rstatus='PARSE_ERROR';reason.append(type(exc).__name__+': '+str(exc)[:100])
        pkey=(r['dataset'],r['source_id'],r['rule_label'],r['time_window'])
        # Historical rule names differ across some display and stored RDF names.
        pname={v:k for k,v in RULE_NAMES.items()}.get(rule,'')
        pitem=prov.get((r['dataset'],r['source_id'],pname,r['time_window']))
        if pitem is None:
            pstatus='NOT_SAMPLED'
        else:
            pstatus='PASS' if abs(float(pitem['score'])-score)<=TOL_SCORE else 'FAIL'
        if rstatus=='FAIL' or vstatus=='FAIL' or pstatus=='FAIL':status='FAIL';auto_score='0'
        elif rstatus=='PASS':status='VERIFIED_RAW';auto_score='2'
        elif vstatus=='PASS':status='VISIBLE_FORMULA_ONLY';auto_score='1'
        else:status='INSUFFICIENT_EVIDENCE';auto_score=''
        if vstatus=='INSUFFICIENT': reason.append('Visible evidence lacks a required rule input')
        if rstatus=='NOT_AVAILABLE':reason.append('Original propagation tree not accessible in this run')
        if pstatus=='NOT_SAMPLED':reason.append('This specific activation not in sampled RDF, not a violation')
        if not reason:reason.append('Formula and available evidence agree within rounding tolerance')
        direction='twitter15_to_twitter16' if r['dataset']=='twitter16' else 'twitter16_to_twitter15'
        s=sens.get((direction,rule),{})
        out.append({'case_id':r['case_id'],'dataset':r['dataset'],'source_id':r['source_id'],
          'window':r['time_window'],'policy':r['tree_policy'],'rule_feature':rule,
          'stratum':r['sampling_stratum'],'stored_rule_score':score,
          'score_recomputed_visible': '' if visible is None else f'{visible:.9f}',
          'score_recomputed_raw':'' if actual is None else f'{actual:.9f}',
          'visible_abs_error':'' if visible is None else f'{abs(visible-score):.9f}',
          'raw_abs_error':'' if actual is None else f'{abs(actual-score):.9f}',
          'visible_status':vstatus,'raw_status':rstatus,
          'observed_feature_count':len(observed),'raw_feature_abs_error_max':'' if not diffs else f'{max(diffs.values()):.9f}',
          'sample_provenance_status':pstatus,'auto_status':status,
          'automated_reproducibility_0_2':auto_score,
          'source_prediction_p_C':k['p_C'],'source_prediction_p_D':k['p_D'],
          'source_prediction_abs_delta_CD':f'{abs(float(k["p_D"])-float(k["p_C"])):.9f}',
          'correctness_D':str(int(k['D_prediction']==k['gold_rumor_label'])),
          'rule_group_mean_abs_p_intervention':s.get('mean_abs_probability_delta',''),
          'case_specific_rule_intervention_performed':'NO',
          'negative_delay_edge_count':issue.get('negative_delay','') if issue else '',
          'backward_edge_count':issue.get('backward_edges','') if issue else '',
          'audit_reason':'; '.join(reason),
          'expert_rating':'NOT_AVAILABLE',
          'semantic_truth_of_source':'NOT_ASSESSED',
          'original_source_text':r['source_text']})
    fieldnames=list(out[0]);writecsv(root/'automated_case_audit_216.csv',out,fieldnames)
    stats,rules=summarize(out);writecsv(root/'rule_level_summary.csv',rules,list(rules[0]))
    stats['original_git_sha']=ORIGINAL_SHA;stats['original_tree_source']=raw_source
    stats['score_tolerance']=TOL_SCORE;stats['input_csv_sha256']=hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()
    (root/'audit_manifest.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    report=['# Automated Ontology Explanation Verification — 216 case audit','',
      '**This is NOT expert validation.** It audits deterministic rule execution and available provenance only.','',
      f"- Cases: **{len(out)}**. Original trees available for {sum(x['raw_status']!='NOT_AVAILABLE' for x in out)} cases.",
      f"- Scores verified against original trees: **{stats['full_raw_trees_verified']}**.",
      f"- Scores recomputed from visible rounded evidence: **{stats['visible_score_verified']}**.",
      f"- Cases with insufficient *visible* evidence: **{sum(x['visible_status']=='INSUFFICIENT' for x in out)}**.",
      f"- Verified mismatches: **{stats['overall_status_counts'].get('FAIL',0)}**.",'',
      '## Interpret scores correctly','',
      '- **2**: rule score and visible evidence agree with original propagation tree.',
      '- **1**: score rederived from rounded displayed features only; source tree not verified.',
      '- **0**: a demonstrable score/evidence/provenance mismatch.',
      '- **blank**: insufficient evidence, NOT a failed rule.',
      '', 'These are **computational reproducibility levels**, not human ratings of semantic validity or usefulness.',
      'No verification of rumor truth, support/deny stance, semantic explanation quality, OWL-DL entailment, or causal faithfulness is claimed.','',
      '## Six rules','',
      '| Rule | Cases | Raw pass | Visible pass | Visible incomplete | Saturated >=.95 |',
      '|---|---:|---:|---:|---:|---:|']
    report += [f"| {x['rule']} | {x['n']} | {x['raw_pass']} | {x['visible_pass']} | {x['insufficient_without_raw']} | {x['score_saturated_ge_095']} |" for x in rules]
    report+=['','## Interpreting model faithfulness','',
      'Absolute p_D−p_C is a **model-comparison** diagnostic, not the causal contribution of the listed single rule.',
      'Any group-level rule sensitivity is from a prior median-substitution analysis of the trained model, not a case-specific counterfactual.',
      'To measure individual-rule intervention faithfully, load the frozen trained model and perturb the original rule feature (and ideally its correlated evidence).',
      '', '## Source provenance','',f'- Original repository SHA: `{ORIGINAL_SHA}`',
      f'- Raw trees: `{raw_source}`','- Automated numeric tolerance: `0.005` (features shown at four decimals).',
      '']
    (root/'AUTOMATED_AUDIT_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    zpath=root.parent/'Ontology216_Automated_Verification_Results.zip'
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file() and 'raw_trees' not in file.parts and file.name!='twitter1516_pinned_full.zip':
                z.write(file,file.relative_to(root))
    print(json.dumps(stats,indent=2,ensure_ascii=False));print('OUTPUT_ZIP',zpath)
    return zpath

if __name__=='__main__':main()
