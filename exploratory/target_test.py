import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
f=pd.read_parquet('artifacts/calls_features.parquet')
l=pd.read_parquet('artifacts/calls_labels.parquet')
d=f.merge(l.drop(columns=['date']),on='call_id')
d['neg_ret']=(d.post_excess_ret<-0.02).astype(int)
d['abs_ret']=(d.post_excess_ret.abs()>=d.post_excess_ret.abs().quantile(2/3)).astype(int)
d['post_high']=(d.post_vol>=d.post_vol.quantile(2/3)).astype(int)
acc=[c for c in d.columns if c.startswith('acc_')]
lex=['n_words','n_unique','ttr','n_chars','n_numbers','hedge_rate','neg_rate','fwd_rate','modal_rate']
meta=['n_segments','audio_len']
tr=d[d.date<'2017-01-01']; te=d[d.date>='2017-07-01']
def lgbm(X,y): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=31,min_child_samples=40,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=4,verbose=-1).fit(X,y)
def lr(X,y): return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced'))]).fit(X,y)
for tgt in ['label','neg_ret','abs_ret','post_high']:
    print('\nTARGET',tgt,'rate',round(te[tgt].mean(),3))
    for name,cols,fn in [('pre_vol',['pre_vol'],lr),('acoustic',acc+meta,lgbm),('lex',lex,lr),('ac+lex',acc+lex+meta,lgbm)]:
        m=fn(tr[cols],tr[tgt]); p=m.predict_proba(te[cols])[:,1]
        print(f'  {name:10s} auc={roc_auc_score(te[tgt],p):.4f}')
