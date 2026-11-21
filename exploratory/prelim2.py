import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
f=pd.read_parquet('artifacts/calls_features.parquet')
dyn=pd.read_parquet('artifacts/call_dynamics.parquet')
dl=pd.read_parquet('artifacts/calls_deltas.parquet')
l=pd.read_parquet('artifacts/calls_labels.parquet')
d=f.merge(dyn,on='call_id').merge(dl,on=['call_id','ticker','date']).merge(l[['call_id','label','vol_ratio','pre_vol']],on='call_id')
d=d[d.days_since_prev.notna()].copy()
acc=[c for c in d.columns if c.startswith('acc_')]; dyn_c=[c for c in d.columns if c.startswith('dyn_')]
lex=['n_words','n_unique','ttr','n_chars','n_numbers','hedge_rate','neg_rate','fwd_rate','modal_rate','qa_start_frac','has_transition']
dacc=[c for c in d.columns if c.startswith('d_acc_')]; ddyn=[c for c in d.columns if c.startswith('d_dyn_')]; dlex=[c for c in d.columns if c.startswith('d_hedge') or c.startswith('d_neg') or c.startswith('d_fwd') or c.startswith('d_modal')]
meta=['n_segments','audio_len','days_since_prev']
tr=d[d.date<'2017-01-01']; te=d[d.date>='2017-07-01']
print('rows',len(d),'train',len(tr),'test',len(te),'rate',round(te.label.mean(),3))
def lgbm(X,y): return lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=40,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,n_jobs=4,verbose=-1).fit(X,y)
def lr(X,y): return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced'))]).fit(X,y)
def ev(name,cols,fn):
    m=fn(tr[cols],tr.label); p=m.predict_proba(te[cols])[:,1]
    print(f'{name:22s} auc={roc_auc_score(te.label,p):.4f} ap={average_precision_score(te.label,p):.4f} brier={brier_score_loss(te.label,p):.4f}')
ev('pre_vol',['pre_vol'],lr)
ev('acoustic levels',acc+meta,lgbm)
ev('acoustic + dyn',acc+dyn_c+meta,lgbm)
ev('acoustic + deltas',acc+dyn_c+dacc+ddyn+meta,lgbm)
ev('lex + dlex',lex+dlex,lr)
ev('all acoustic+lex+d',acc+dyn_c+dacc+ddyn+lex+dlex+meta,lgbm)
m=lgbm(tr[acc+dyn_c+dacc+ddyn+lex+dlex+meta],tr.label)
imp=pd.DataFrame({'f':acc+dyn_c+dacc+ddyn+lex+dlex+meta,'gain':m.booster_.feature_importance('gain')}).sort_values('gain',ascending=False)
print(imp.head(20).to_string(index=False))
