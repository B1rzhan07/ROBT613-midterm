"""Create presentation inputs directly from notebook tables, retaining numeric lineage."""
import json
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1];r=root/'results'
m=json.loads((r/'run_manifest.json').read_text());assert m['full_planned_cohort']
read=lambda n:pd.read_csv(r/f'{n}.csv',index_col=0)
c=read('comparison_subjects'); tests=read('hypothesis_tests');sens=read('sensitivity_tests');erp=read('erp')
obj=dict(manifest=m,subjects=c.reset_index().to_dict('records'),tests=tests.reset_index().rename(columns={'index':'hypothesis'}).iloc[:3].to_dict('records'),
 sensitivity=sens.reset_index().rename(columns={'index':'hypothesis'}).to_dict('records'),
 means={k:float(c[k].mean()) for k in ['within','loso_matched','loso']},
 sds={k:float(c[k].std(ddof=1)) for k in ['within','loso_matched','loso']},
 median_gap=float((c.within-c.loso).median()),median_fold_sd=float(c.within_fold_sd.median()),
 positive_erp=int((erp.Pz_window_mean_uV>0).sum()),mean_erp=float(erp.Pz_window_mean_uV.mean()))
(r/'presentation_data.json').write_text(json.dumps(obj,indent=2,allow_nan=False))
print('Wrote presentation_data.json')
