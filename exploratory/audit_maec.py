import re, pandas as pd, numpy as np
from pathlib import Path
root = Path('data/maec/MAEC_Dataset')
rows=[]
def sent_count(t):
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9$"“])', t)
    return len([p for p in parts if p.strip()])
for d in sorted(root.iterdir()):
    if not d.is_dir(): continue
    m=re.match(r'(\d{8})_([A-Z0-9.\-]+)$', d.name)
    if not m: continue
    date, ticker = m.groups()
    txt=(d/'text.txt').read_text(errors='ignore')
    df=pd.read_csv(d/'features.csv')
    qa = bool(re.search(r'question.and.answer|q\s*&\s*a|question-and-answer', txt, re.I))
    rows.append(dict(call_id=d.name, date=date, ticker=ticker, n_rows=len(df), n_sent=sent_count(txt),
                     n_chars=len(txt), has_qa=qa, n_null=int(df.isna().sum().sum()),
                     audio_len=float(pd.to_numeric(df.iloc[:,-1], errors='coerce').sum())))
a=pd.DataFrame(rows)
a['diff']=a.n_rows-a.n_sent
print('shape', a.shape)
print(a[['n_rows','n_sent','diff','n_chars','audio_len']].describe().round(1).to_string())
print('has_qa', round(a.has_qa.mean(),3))
print('date range', a.date.min(), a.date.max())
print('unique tickers', a.ticker.nunique())
print('abs diff<=2', round((a['diff'].abs()<=2).mean(),3), 'abs diff<=5', round((a['diff'].abs()<=5).mean(),3))
print('null rows', int((a.n_null>0).sum()))
a.to_csv('artifacts/maec_audit.csv', index=False)
print(a.sort_values('diff').head(3)[['call_id','n_rows','n_sent','diff']].to_string(index=False))
print(a.sort_values('diff').tail(3)[['call_id','n_rows','n_sent','diff']].to_string(index=False))
