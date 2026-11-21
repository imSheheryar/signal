import pandas as pd, numpy as np
f=pd.read_parquet('artifacts/calls_features.parquet')
dyn=pd.read_parquet('artifacts/call_dynamics.parquet')
f=f.merge(dyn,on='call_id',how='left')
f=f.sort_values(['ticker','date']).reset_index(drop=True)
num=[c for c in f.columns if c.startswith(('acc_','dyn_','hedge_','neg_','fwd_','modal_'))]
g=f.groupby('ticker',sort=False)
prev_date=g['date'].shift(1)
f['days_since_prev']=(f['date']-prev_date).dt.days
out=f[['call_id','ticker','date','days_since_prev']].copy()
for c in num:
    out[f'd_{c}']=f[c]-g[c].shift(1)
out.to_parquet('artifacts/calls_deltas.parquet',index=False)
print(out.shape, 'with prev', out.days_since_prev.notna().sum())
print(out.days_since_prev.describe().round(1).to_string())
