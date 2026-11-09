"""Static smoke test for project files and code (does not train)."""
from pathlib import Path
import ast,csv,json
r=Path(__file__).resolve().parent.parent
py=list((r/'src').rglob('*.py'))+list((r/'scripts').rglob('*.py'))
for p in py:ast.parse(p.read_text(encoding='utf-8'))
for p in (r/'notebooks').rglob('*.ipynb'):
    nb=json.loads(p.read_text(encoding='utf-8'))
    assert nb['cells']
print('PASS:',len(py),'Python code files and notebook JSON verified')
