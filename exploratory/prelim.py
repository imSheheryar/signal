import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
f=pd.read_parquet('artifacts/calls_features.parquet')
l=pd.read_parquet('artifacts/calls_labels.parquet')
d=f.merge(l[['call_id','label','vol_ratio','pre_vol']],on='call_id')
acc=[c for c in d.columns if c.startswith('acc_')]
lex=['n_words','n_unique','ttr','n_chars','n_numbers','hedge_rate','neg_rate','fwd_rate','modal_rate','qa_start_frac','has_transition']
meta=['n_segments','audio_len']
tr=d[d.date<'2017-01-01']; va=d[(d.date>='2017-01-01')&(d.date<'2017-07-01')]; te=d[d.date>='2017-07-01']
print('rows',len(d),'split',len(tr),len(va),len(te),'rate',round(te.label.mean(),3))
def lgbm(X,y): return lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=40,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,n_jobs=4,verbose=-1).fit(X,y)
def lr(X,y): return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced'))]).fit(X,y)
def ev(name,m,cols):
    p=m.predict_proba(te[cols])[:,1]
    print(f'{name:12s} auc={roc_auc_score(te.label,p):.4f} ap={average_precision_score(te.label,p):.4f} brier={brier_score_loss(te.label,p):.4f}')
for name,cols,fn in [('pre_vol',['pre_vol'],lr),('acoustic',acc+meta,lgbm),('lexicon',lex,lr),('ac+lex',acc+lex+meta,lgbm)]:
    m=fn(tr[cols],tr.label); ev(name,m,cols)
# feature importance
m=lgbm(tr[acc+lex+meta],tr.label)
imp=pd.DataFrame({'f':acc+lex+meta,'gain':m.booster_.feature_importance('gain')}).sort_values('gain',ascending=False)
print(imp.head(15).to_string(index=False))
