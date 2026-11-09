"""Pinned-source Twitter-COVID19 archive ingestion and schema/overlap preflight.
No test-label driven tuning. Does not claim external model evaluation.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,os,re,sys,zipfile,unicodedata
from pathlib import Path
from urllib.request import Request,urlopen

REPO='DanielLin97/ACLR4RUMOR-NAACL2022'
SHA='9182537d596f994119ad10bf750f726189e1c38d'
URL=f'https://raw.githubusercontent.com/{REPO}/{SHA}/data/Twitter/COVID-19/Twitter-COVID19.7z'
EXPECTED={'events':400,'rumor':252,'non_rumor':148}

def norm(s):
 s=unicodedata.normalize('NFKC',str(s)).lower()
 s=re.sub(r'https?://\S+|www\.\S+',' URL ',s)
 s=re.sub(r'@[\w_]+',' USER ',s)
 return re.sub(r'[^\w]+',' ',s,flags=re.UNICODE).strip()

def sha256(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  while True:
   b=f.read(4*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()

def download_archive(dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and dest.stat().st_size>2_000_000:return
 req=Request(URL,headers={'User-Agent':'academic-external-validation/1.0'})
 with urlopen(req,timeout=180) as r, open(dest,'wb') as f:
  while True:
   b=r.read(1024*1024)
   if not b:break
   f.write(b)
 if dest.stat().st_size<2_000_000:raise RuntimeError('Archive unexpectedly small; inspect network/download response.')

def archive_extract(archive,dst):
 try:import py7zr
 except ImportError:raise RuntimeError('py7zr unavailable; Kaggle setup cell must install py7zr')
 dst.mkdir(parents=True,exist_ok=True)
 with py7zr.SevenZipFile(str(archive),'r') as z:
  names=z.getnames()
  unsafe=[n for n in names if n.startswith('/') or '..' in Path(n).parts]
  if unsafe:raise RuntimeError(f'Unsafe archive member names: {unsafe[:3]}')
  z.extractall(path=str(dst))
 return names

def inspect_files(root,out):
 candidates=[];extensions=collections.Counter()
 for p in root.rglob('*'):
  if not p.is_file():continue
  rel=str(p.relative_to(root));extensions[p.suffix.lower()]+=1
  if p.suffix.lower() not in {'.txt','.csv','.tsv','.json','.jsonl','.tree'}:continue
  try:
   with p.open(encoding='utf-8-sig',errors='replace') as f:
    first=[f.readline().rstrip('\n\r')[:600] for _ in range(3)]
  except Exception as e:first=[f'READ_ERROR {str(e)[:120]}']
  candidates.append({'file':rel,'bytes':p.stat().st_size,'head_lines':first})
 candidates.sort(key=lambda x:x['file'])
 return candidates,dict(extensions)

def try_parse_labels(candidates,base):
 found=[]
 for entry in candidates:
  if 'label' not in entry['file'].lower() or entry['bytes']>6_000_000:continue
  path=base/entry['file']
  try:
   lines=path.read_text(encoding='utf-8-sig',errors='replace').splitlines();y={}; bad=0
   for x in lines:
    parts=re.split(r'\t|[:, ]+',x.strip(),maxsplit=2)
    parts=[p.strip().strip("'\"") for p in parts if p.strip()]
    if len(parts)<2:bad+=1;continue
    matched=[p.lower() for p in parts if p.lower() in {'rumor','rumour','non-rumor','nonrumor','non-rumour','nonrumour','0','1','true','false'}]
    if not matched:bad+=1;continue
    # Determine id via non-label field; do not assume ID is numeric.
    label=matched[0];identifier=next((p for p in parts if p.lower()!=label),None)
    if identifier:y[identifier]=label
   found.append({'file':entry['file'],'rows':len(lines),'parseable_labels':len(y),'other_rows':bad,'label_distribution':dict(collections.Counter(y.values()))})
  except Exception as e:found.append({'file':entry['file'],'error':str(e)[:200]})
 return found

def try_parse_text_and_graph(candidates):
 def looks_parent(edges):return any('->' in l or re.search(r'\bparent\b|\bsource\b|\broot\b',l,re.I) for e in edges for l in e['head_lines'])
 likely=[e for e in candidates if ('tree' in e['file'].lower() or 'graph' in e['file'].lower() or 'data' in e['file'].lower())]
 datafiles=[e for e in likely if any('\t' in x or ',' in x for x in e['head_lines'])]
 time_fields=[e['file'] for e in candidates if any(re.search(r'(delay|created_at|timestamp|post.time|time_offset)',s,re.I) for s in e['head_lines'])]
 return {'tree_path_count':sum('/tree/' in e['file'].lower() or '/trees/' in e['file'].lower() for e in candidates),
         'candidate_graph_text_files':[e['file'] for e in likely[:30]],
         'tabular_data_files':[e['file'] for e in datafiles[:30]],
         'time_field_candidates':time_fields[:30],
         'has_graph_indicator':looks_parent(likely),
         'timestamp_semantics_confirmed':False}

def overlap_audit(out,candidates,root,ref_path):
 if not ref_path or not Path(ref_path).exists():return {'status':'REFERENCE_NOT_PROVIDED'}
 with open(ref_path,encoding='utf-8-sig',newline='') as f:ref=list(csv.DictReader(f))
 ids={str(x.get('id','')).strip() for x in ref}
 texts={norm(x.get('source_text','')) for x in ref if norm(x.get('source_text',''))}
 matches=[]
 # preflight can only check obvious tabular source files; never claim complete overlap audit
 for c in candidates:
  if c['bytes']>15_000_000 or not any(z in c['file'].lower() for z in ['source','data_all']):continue
  path=root/c['file']
  with path.open(encoding='utf-8-sig',errors='replace') as f:
   for line_no,line in enumerate(f,1):
    if line_no>100000:break
    vals=line.strip().split('\t')
    if not vals:continue
    if vals[0] in ids:matches.append({'file':c['file'],'line_no':line_no,'type':'ID','value':vals[0]})
    if len(vals)>1 and norm(vals[-1]) in texts:matches.append({'file':c['file'],'line_no':line_no,'type':'EXACT_NORMALIZED_TEXT','value':norm(vals[-1])[:100]})
 with open(out/'overlap_obvious_matches.csv','w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['file','line_no','type','value']);w.writeheader();w.writerows(matches)
 return {'status':'PRELIMINARY_ONLY','reference_rows':len(ref),'matching_line_occurrences':len(matches), 'limitation':'not full event-level overlap/near-duplicate audit until target schema verified'}

def main(args):
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 extraction=out/'extracted'
 if args.extracted_dir:
  extraction=Path(args.extracted_dir);archive=None;names=[]
 else:
  archive=Path(args.archive) if args.archive else out/'Twitter-COVID19.7z'
  download_archive(archive) if not archive.exists() else None
  names=archive_extract(archive,extraction)
 files,extensions=inspect_files(extraction,out)
 labels=try_parse_labels(files,extraction)
 graph=try_parse_text_and_graph(files)
 overlap=overlap_audit(out,files,extraction,args.source_reference)
 report={'dataset':'Twitter-COVID19','source_repository':REPO,'pinned_commit':SHA,'source_url':URL,
         'archive_sha256':sha256(archive) if archive else None,'archive_members':len(names),
         'files_scanned':len(files),'extension_counts':extensions,'expected_paper_counts':EXPECTED,
         'label_file_candidates':labels,'graph_and_time_preflight':graph,'preliminary_overlap':overlap,
         'external_validation_completed':False,
         'external_validation_ready':False,
         'next_decision':'Verify 400 distinct event IDs, binary label mapping, source text, parent-child edges, and real timestamps. If any is missing, do NOT run temporal C-D with invented values.',
         'limitations':'The archive schema has not yet been independently inspected in the model container. This is download/data preflight, NOT C-D external validation.'}
 (out/'PREFLIGHT_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 (out/'ARCHIVE_CONTENTS_PREVIEW.json').write_text(json.dumps(files[:100],indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['files_scanned','extension_counts','label_file_candidates','graph_and_time_preflight','external_validation_completed']},indent=2,ensure_ascii=False))
 outzip=out.parent/'TwitterCOVID19_External_Preflight_Results.zip'
 with zipfile.ZipFile(outzip,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['PREFLIGHT_REPORT.json','ARCHIVE_CONTENTS_PREVIEW.json','overlap_obvious_matches.csv']:
   p=out/name
   if p.exists():z.write(p,arcname=name)
 print('Saved',outzip)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive');p.add_argument('--extracted-dir');p.add_argument('--source-reference');p.add_argument('--output',default='/kaggle/working/twitter_covid19_preflight')
 main(p.parse_args())
