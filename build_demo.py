import pandas as pd, numpy as np, re
from pathlib import Path
A=Path('artifacts'); R=Path('data/maec/MAEC_Dataset')
pred=pd.read_parquet(A/'test_predictions.parquet')
shap=pd.read_parquet(A/'shap_test.parquet')
f=pd.read_parquet(A/'calls_features.parquet')
dyn=pd.read_parquet(A/'call_dynamics.parquet')
txt=pd.read_parquet(A/'calls_text.parquet')
emb_path=A/'text_embeddings.parquet'
emb=pd.read_parquet(emb_path) if emb_path.exists() else pd.DataFrame({'call_id':[]})
base=f.merge(dyn,on='call_id',how='left').merge(txt,on='call_id',how='left').merge(pred[['call_id','pre_vol']],on='call_id',how='left')
if len(emb): base=base.merge(emb,on='call_id',how='left')
calls=pred.merge(base,on='call_id',how='left',suffixes=('','_feat'))
# choose display probability: best late fusion column if present
calls['risk']=calls['pred_late_pre_text_ac']
calls['risk_pct']=calls.risk.rank(pct=True)
calls['outcome']=np.where(calls.y==1,'High volatility','Normal')
# readable snippet
def snip(t):
    t=re.sub(r'\s+',' ',t).strip()
    return t if len(t)<=1800 else t[:900]+' ... '+t[-900:]
calls['snippet']=calls.text.map(snip)
keep=['call_id','ticker','date','y','outcome','post_vol','pre_vol','risk','risk_pct']+[c for c in ['pred_pre_vol','pred_text_tfidf','pred_acoustic','pred_fusion','pred_late_pre_text','pred_late_pre_text_ac','pred_late_all'] if c in calls.columns]
# avoid duplicate pre_vol names
keep=list(dict.fromkeys(keep))
demo=calls[keep+['snippet']].copy()
demo.to_parquet(A/'demo_calls.parquet',index=False)
# drivers: non-SVD, non-embedding features only
shap_cols=[c for c in shap.columns if c!='call_id']
readable=[c for c in shap_cols if not c.startswith(('svd_','emb_'))]
rows=[]
for _,r in shap.iterrows():
    vals=base[base.call_id==r.call_id]
    if len(vals)==0: continue
    vals=vals.iloc[0]
    contribs=[(c,float(r[c]),float(vals[c]) if c in vals and pd.notna(vals[c]) else np.nan) for c in readable]
    contribs.sort(key=lambda x:abs(x[1]),reverse=True)
    for c,v,val in contribs[:6]:
        rows.append(dict(call_id=r.call_id,feature=c,contribution=v,value=val))
pd.DataFrame(rows).to_parquet(A/'demo_drivers.parquet',index=False)
# prosody series for test calls
key=['Mean pitch','Mean intensity','Jitter local','Shimmer local','Mean HNR','Fraction of unvoiced']
pros=[]
for cid in demo.call_id:
    p=R/cid/'features.csv'
    if not p.exists(): continue
    d=pd.read_csv(p).apply(pd.to_numeric,errors='coerce')
    d=d.reset_index().rename(columns={'index':'segment'})
    for c in key:
        if c in d.columns:
            for _,row in d.iterrows():
                pros.append(dict(call_id=cid,segment=int(row['segment']),metric=c,value=float(row[c]) if pd.notna(row[c]) else np.nan))
pd.DataFrame(pros).to_parquet(A/'demo_prosody.parquet',index=False)
print('demo',demo.shape,'drivers',len(rows),'prosody',len(pros))
print(demo.head(3).to_string())
