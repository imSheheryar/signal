import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
f=pd.read_parquet('artifacts/calls_features.parquet'); t=pd.read_parquet('artifacts/calls_text.parquet'); l=pd.read_parquet('artifacts/calls_labels.parquet')
d=f.merge(t,on='call_id').merge(l[['call_id','post_vol','pre_vol']],on='call_id')
d['y']=(d.post_vol>=d.post_vol.quantile(2/3)).astype(int)
tr=d[d.date<'2017-01-01']; te=d[d.date>='2017-07-01']
vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=3,stop_words='english',sublinear_tf=True)
Xtr=vec.fit_transform(tr.text); Xte=vec.transform(te.text)
for name,Xtr2,Xte2 in [('text',Xtr,Xte),('text+pre',hstack([Xtr,csr_matrix(tr[['pre_vol']].values)]).tocsr(),hstack([Xte,csr_matrix(te[['pre_vol']].values)]).tocsr())]:
    m=LogisticRegression(max_iter=3000,C=0.5,class_weight='balanced').fit(Xtr2,tr.y)
    print(name, round(roc_auc_score(te.y,m.predict_proba(Xte2)[:,1]),4))
m=LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced').fit(tr[['pre_vol']],tr.y)
print('pre only',round(roc_auc_score(te.y,m.predict_proba(te[['pre_vol']])[:,1]),4))
