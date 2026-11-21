import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
f=pd.read_parquet('artifacts/calls_features.parquet'); t=pd.read_parquet('artifacts/calls_text.parquet'); l=pd.read_parquet('artifacts/calls_labels.parquet')
d=f.merge(t,on='call_id').merge(l[['call_id','label','vol_ratio','pre_vol']],on='call_id')
d['log_ratio']=np.log(d.vol_ratio.clip(0.05,10)); d['log_pre']=np.log(d.pre_vol.clip(0.02,2))
tr=d[d.date<'2017-01-01']; te=d[d.date>='2017-07-01']
reg=LinearRegression().fit(tr[['log_pre']],tr.log_ratio)
d['resid']=d.log_ratio-reg.predict(d[['log_pre']])
q=d.loc[d.date<'2017-01-01','resid'].quantile(2/3)
d['res_label']=(d.resid>=q).astype(int)
tr=d[d.date<'2017-01-01']; te=d[d.date>='2017-07-01']
print('res label rate train/test',round(tr.res_label.mean(),3),round(te.res_label.mean(),3),'q',round(q,3))
acc=[c for c in d.columns if c.startswith('acc_')]; lex=['n_words','n_unique','ttr','n_chars','n_numbers','hedge_rate','neg_rate','fwd_rate','modal_rate']; meta=['n_segments','audio_len']
def lgbm(X,y): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=31,min_child_samples=40,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=4,verbose=-1).fit(X,y)
def lr(X,y): return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced'))]).fit(X,y)
for name,cols,fn in [('pre_vol',['pre_vol'],lr),('acoustic',acc+meta,lgbm),('lex',lex,lr),('ac+lex',acc+lex+meta,lgbm)]:
    m=fn(tr[cols],tr.res_label); p=m.predict_proba(te[cols])[:,1]
    print(f'{name:10s} auc={roc_auc_score(te.res_label,p):.4f}')
vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=3,stop_words='english',sublinear_tf=True)
Xtr=vec.fit_transform(tr.text); Xte=vec.transform(te.text)
m=LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced').fit(Xtr,tr.res_label)
print('tfidf     auc=',round(roc_auc_score(te.res_label,m.predict_proba(Xte)[:,1]),4))
