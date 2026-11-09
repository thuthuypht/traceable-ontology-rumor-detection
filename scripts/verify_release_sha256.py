"""Verify packaged public files against manifest checksums."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
f=root/'release_manifest_sha256.json'
man=json.loads(f.read_text(encoding='utf-8'))
bad=[]
for rel,expected in man['files'].items():
    target=root/rel
    if not target.is_file():bad.append((rel,'MISSING'));continue
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    if sha!=expected:bad.append((rel,'SHA256 MISMATCH'))
print('Checked',len(man['files']),'public files; failures',len(bad))
for x in bad:print(x)
if bad:raise SystemExit(1)
