import re, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path('data/maec/MAEC_Dataset')
KEY=['Mean pitch','Mean intensity','Jitter local','Shimmer local','Mean HNR','Fraction of unvoiced','Degree of voice breaks','Mean autocorrelation']
rows=[]
for d in sorted(ROOT.iterdir()):
    if not d.is_dir(): continue
    m=re.match(r'(\d{8})_([A-Z0-9.\-]+)$', d.name)
    if not m: continue
    df=pd.read_csv(d/'features.csv').apply(pd.to_numeric, errors='coerce')
    n=len(df); x=np.linspace(0,1,n) if n>1 else np.array([0.0])
    rec={'call_id':d.name}
    for c in KEY:
        if c not in df.columns: continue
        v=df[c].to_numpy(dtype=float)
        ok=np.isfinite(v)
        if ok.sum()>=3:
            rec[f'dyn_{c}_slope']=float(np.polyfit(x[ok],v[ok],1)[0])
            k=max(1,n//3)
            rec[f'dyn_{c}_delta13']=float(np.nanmean(v[-k:])-np.nanmean(v[:k]))
            rec[f'dyn_{c}_diffstd']=float(np.nanstd(np.diff(v)))
        else:
            rec[f'dyn_{c}_slope']=np.nan; rec[f'dyn_{c}_delta13']=np.nan; rec[f'dyn_{c}_diffstd']=np.nan
    rows.append(rec)
out=pd.DataFrame(rows)
out.to_parquet('artifacts/call_dynamics.parquet',index=False)
print(out.shape); print(out.head(2).to_string())
