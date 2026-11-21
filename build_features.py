import re, json
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path('data/maec/MAEC_Dataset')
OUT = Path('artifacts'); OUT.mkdir(exist_ok=True)

HEDGES = {'approximately','about','around','believe','expect','anticipate','estimate','forecast','intend','may','might','could','should','would','potential','possibly','likely','uncertain','uncertainty','assume','assumption','guidance','outlook','subject to','risk','risks'}
MODALS = {'may','might','could','should','would','will','can'}
NEG = {'not','no','never','none','nothing','without','fail','failed','decline','declined','decrease','decreased','loss','losses','weak','weakness','risk','risks','concern','concerns','challenging','difficult','lower','reduced','impairment','litigation','investigation'}
FWD = {'will','expect','anticipate','forecast','guidance','outlook','going forward','next quarter','next year','future','plan','intend','target'}

TRANSITION = re.compile(r'(question[-\s]and[-\s]answer|q\s*&\s*a|turn(?:ing)? (?:the call|it) (?:back )?over to the operator|open(?:ing)? (?:up )?the lines|now (?:take|begin) (?:your |the )?questions|proceed to the question|question session)', re.I)

def text_features(t):
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", t.lower())
    n = max(len(words), 1)
    wset = set(words)
    nums = re.findall(r'\$?\d[\d,\.]*%?', t)
    hedges = sum(1 for w in words if w in HEDGES)
    neg = sum(1 for w in words if w in NEG)
    fwd = sum(1 for w in words if w in FWD)
    modal = sum(1 for w in words if w in MODALS)
    m = TRANSITION.search(t)
    qa_frac = (m.start()/len(t)) if m else np.nan
    return dict(n_words=len(words), n_unique=len(wset), ttr=len(wset)/n, n_chars=len(t),
                n_numbers=len(nums), hedge_rate=hedges/n, neg_rate=neg/n, fwd_rate=fwd/n,
                modal_rate=modal/n, qa_start_frac=qa_frac, has_transition=int(bool(m)))

rows=[]; texts=[]
for d in sorted(ROOT.iterdir()):
    if not d.is_dir(): continue
    m=re.match(r'(\d{8})_([A-Z0-9.\-]+)$', d.name)
    if not m: continue
    date, ticker = m.groups()
    df=pd.read_csv(d/'features.csv')
    num=df.apply(pd.to_numeric, errors='coerce')
    agg={}
    for c in num.columns:
        s=num[c]
        agg[f'acc_{c}_mean']=s.mean(); agg[f'acc_{c}_std']=s.std()
        agg[f'acc_{c}_min']=s.min(); agg[f'acc_{c}_max']=s.max()
    txt=(d/'text.txt').read_text(errors='ignore')
    tf=text_features(txt)
    rec=dict(call_id=d.name, date=date, ticker=ticker, n_segments=len(df),
             audio_len=float(num.iloc[:,-1].sum()), **agg, **tf)
    rows.append(rec)
    texts.append(dict(call_id=d.name, text=txt))

f=pd.DataFrame(rows)
f['date']=pd.to_datetime(f['date'], format='%Y%m%d')
f.to_parquet(OUT/'calls_features.parquet', index=False)
pd.DataFrame(texts).to_parquet(OUT/'calls_text.parquet', index=False)
print('features', f.shape)
print('date range', f.date.min().date(), f.date.max().date())
print('tickers', f.ticker.nunique())
print('qa transition found', round(f.has_transition.mean(),3))
print(f[['n_words','n_segments','audio_len','hedge_rate','neg_rate','qa_start_frac']].describe().round(3).to_string())
