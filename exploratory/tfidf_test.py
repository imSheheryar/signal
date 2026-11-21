import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import hstack
f=pd.read_parquet('artifacts/calls_features.parquet')
t=pd.read_parquet('artifacts/calls_text.parquet')
l=pd.read_parquet('artifacts/calls_labels.parquet')
d=f.merge(t,on='call_id').merge(l[['call_id','label','pre_vol']],on='call_id')
tr=d[d.date<'2017-01-01']; te=d[d.date>='2017-07-01']
vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=3,stop_words='english',sublinear_tf=True)
Xtr=vec.fit_transform(tr.text); Xte=vec.transform(te.text)
for C in [0.1,0.5,1.0]:
    m=LogisticRegression(max_iter=2000,C=C,class_weight='balanced').fit(Xtr,tr.label)
    p=m.predict_proba(Xte)[:,1]
    print('C',C,'auc',round(roc_auc_score(te.label,p),4),'ap',round(average_precision_score(te.label,p),4))
# top coefficients
m=LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced').fit(Xtr,tr.label)
co=pd.DataFrame({'term':vec.get_feature_names_out(),'coef':m.coef_[0]}).sort_values('coef')
print('negative\n',co.head(15).to_string(index=False)); print('positive\n',co.tail(15).to_string(index=False))
