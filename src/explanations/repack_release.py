"""Finalize the one-run Kaggle package after saved-model reload verification."""
from pathlib import Path
import argparse,shutil
from verify_release import verify
import full_provenance_faithfulness as main

def finish(root:Path,src:Path,dest:Path):
    report=verify(root)
    assert report['failure_count']==0 and report['verified_cases']==216 and report['verified_interventions']==1296
    folder=root/'code';folder.mkdir(exist_ok=True)
    for p in src.iterdir():
        if p.is_file():shutil.copy2(p,folder/p.name)
    main.package(root,dest)
    print('VERIFIED_FINAL_RESULT',dest,'models:',len(list((root/'models').rglob('*.cbm'))),'preprocessors:',len(list((root/'preprocessors').rglob('*.joblib'))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--src',required=True);p.add_argument('--dest',required=True);a=p.parse_args()
    finish(Path(a.root),Path(a.src),Path(a.dest))
