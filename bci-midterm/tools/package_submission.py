"""Package the notebook, evidence and reviewable submission artifacts without EEG data."""
import json,zipfile,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'results/run_manifest.json').read_text());assert manifest['full_planned_cohort']
files=[root/p for p in ['notebooks/project.ipynb','output/pdf/report.pdf','output/pdf/presentation.pdf','output/presentation.pptx','output/SPEAKER_NOTES.md','README.md','CONTRIBUTIONS.md','SUBMISSION_CHECKLIST.md','REPRODUCIBILITY_NOTES.md','requirements.txt','requirements-lock.txt','package.json','package-lock.json','notebook_src/project.py','report/main.tex']]
files += list((root/'tools').glob('*.py'))+list((root/'tools').glob('*.mjs'))
files += list((root/'results').glob('*.csv'))+list((root/'results').glob('*.json'))
files += list((root/'report/images').glob('*.png'))
assert all(f.is_file() for f in files)
package=root/'output/submission.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(set(files)):z.write(p,p.relative_to(root))
with zipfile.ZipFile(package) as z:
 assert z.testzip() is None
print('Packaged',len(set(files)),'files:',package)
print('SHA-256',hashlib.sha256(package.read_bytes()).hexdigest())
