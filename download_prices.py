import pandas as pd, yfinance as yf, time, sys
from pathlib import Path
f=pd.read_parquet('artifacts/calls_features.parquet')
tickers=sorted(f.ticker.unique())
print('tickers', len(tickers), flush=True)
frames=[]
for i in range(0,len(tickers),80):
    batch=tickers[i:i+80]
    yt=[t.replace('.','-') for t in batch]
    try:
        d=yf.download(yt, start='2014-10-01', end='2018-08-01', auto_adjust=True, progress=False, threads=True, group_by='column')
        if isinstance(d.columns, pd.MultiIndex):
            close=d['Close'].copy()
        else:
            close=d[['Close']].copy(); close.columns=batch
        close.columns=[batch[j] for j in range(len(close.columns))]
        frames.append(close)
        print(i, 'ok', close.shape, flush=True)
    except Exception as e:
        print(i, 'ERR', repr(e)[:120], flush=True)
    time.sleep(1)
out=pd.concat(frames, axis=1)
out=out.loc[:,~out.columns.duplicated()]
out.to_parquet('artifacts/prices_close.parquet')
print('saved', out.shape, flush=True)
