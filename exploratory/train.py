import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import joblib

OUT=Path('artifacts')
f=pd.read_parquet(OUT/'calls_features.parquet')
lab=pd.read_parquet(OUT/'calls_labels.parquet')
emb=pd.read_parquet(OUT/'text_embeddings.parquet')
df=f.merge(lab[['call_id','label','vol_ratio','pre_vol','post_excess_ret']], on='call_id', how='inner').merge(emb, on='call_id', how='inner')
df=df.sort_values('date').reset_index(drop=True)
print('modelling rows', len(df), 'dates', df.date.min().date(), df.date.max().date(), 'label rate', round(df.label.mean(),3))

acc=[c for c in df.columns if c.startswith('acc_')]
lex=['n_words','n_unique','ttr','n_chars','n_numbers','hedge_rate','neg_rate','fwd_rate','modal_rate','qa_start_frac','has_transition']
sent=[c for c in df.columns if c.startswith('sent_')]
embc=[c for c in df.columns if c.startswith('emb_')]
meta=['n_segments','audio_len']
for c in acc+lex+sent+embc+meta:
    df[c]=pd.to_numeric(df[c], errors='coerce')

train=df[df.date<'2017-01-01']; val=df[(df.date>='2017-01-01')&(df.date<'2017-07-01')]; test=df[df.date>='2017-07-01']
print('split', len(train), len(val), len(test), 'test rate', round(test.label.mean(),3))

def fit_lr(X,y):
    return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(max_iter=3000,C=0.5,class_weight='balanced'))]).fit(X,y)
def fit_lgb(X,y):
    return lgb.LGBMClassifier(n_estimators=500,learning_rate=0.03,num_leaves=31,min_child_samples=40,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,n_jobs=4,verbose=-1).fit(X,y)
def p(m,X): return m.predict_proba(X)[:,1]
def met(y,pr):
    return dict(auc=roc_auc_score(y,pr), ap=average_precision_score(y,pr), brier=brier_score_loss(y,pr), logloss=log_loss(y,pr))

def boot(y,pr,n=1000):
    rng=np.random.default_rng(42); out={'auc':[],'ap':[]}
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        out['auc'].append(roc_auc_score(y[i],pr[i])); out['ap'].append(average_precision_score(y[i],pr[i]))
    return {k:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) for k,v in out.items()}

models={}
# pre-call volatility baseline (mean reversion, no call content)
models['pre_vol']=fit_lr(train[['pre_vol']],train.label)
# text-only: sentiment + lexicon
models['text_lex']=fit_lr(train[lex+sent],train.label)
# text embeddings PCA + LR
pca=PCA(n_components=50,random_state=42).fit(train[embc].fillna(0))
Xtr=pca.transform(train[embc].fillna(0)); Xva=pca.transform(val[embc].fillna(0)); Xte=pca.transform(test[embc].fillna(0))
models['text_emb']=fit_lr(Xtr,train.label)
# acoustic
models['acoustic']=fit_lgb(train[acc+meta],train.label)
# fusion
models['fusion']=fit_lgb(train[acc+lex+sent+embc+meta],train.label)

rows=[]; preds={}
for name,m in models.items():
    if name=='pre_vol': pv,pt=p(m,val[['pre_vol']]),p(m,test[['pre_vol']])
    elif name=='text_emb': pv,pt=p(m,Xva),p(m,Xte)
    elif name=='text_lex': pv,pt=p(m,val[lex+sent]),p(m,test[lex+sent])
    elif name=='acoustic': pv,pt=p(m,val[acc+meta]),p(m,test[acc+meta])
    else: pv,pt=p(m,val[acc+lex+sent+embc+meta]),p(m,test[acc+lex+sent+embc+meta])
    # isotonic calibration on validation
    iso=IsotonicRegression(out_of_bounds='clip').fit(pv,val.label)
    pt_cal=iso.predict(pt)
    mtr=met(test.label,pt); mcal=met(test.label,pt_cal)
    ci=boot(test.label.values,pt)
    rows.append(dict(model=name, **{f'test_{k}':v for k,v in mtr.items()}, **{f'cal_{k}':v for k,v in mcal.items()},
                     auc_lo=ci['auc'][0], auc_hi=ci['auc'][1], ap_lo=ci['ap'][0], ap_hi=ci['ap'][1]))
    preds[name]=pt_cal
# late fusion stack on validation
from sklearn.linear_model import LogisticRegression as LR
stack=LR().fit(np.column_stack([p(models['text_lex'],val[lex+sent]), p(models['acoustic'],val[acc+meta])]), val.label)
pt_stack=stack.predict_proba(np.column_stack([p(models['text_lex'],test[lex+sent]), p(models['acoustic'],test[acc+meta])]))[:,1]
rows.append(dict(model='late_fusion', **{f'test_{k}':v for k,v in met(test.label,pt_stack).items()}, **{f'cal_{k}':v for k,v in met(test.label,pt_stack).items()}, auc_lo=np.nan,auc_hi=np.nan,ap_lo=np.nan,ap_hi=np.nan))
preds['late_fusion']=pt_stack
res=pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
res.to_csv(OUT/'metrics.csv',index=False)
pd.DataFrame({'call_id':test.call_id.values,'y':test.label.values, **{k:v for k,v in preds.items()}}).to_parquet(OUT/'test_predictions.parquet',index=False)
fi=pd.DataFrame({'feature':acc+lex+sent+embc+meta,'gain':models['fusion'].booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False)
fi.to_csv(OUT/'feature_importance.csv',index=False)
print(fi.head(20).to_string(index=False))
# group CV robustness
g=GroupKFold(n_splits=5); aucs=[]
for tr,te in g.split(df,df.label,groups=df.ticker):
    m=fit_lgb(df.iloc[tr][acc+lex+sent+embc+meta],df.iloc[tr].label)
    aucs.append(roc_auc_score(df.iloc[te].label,p(m,df.iloc[te][acc+lex+sent+embc+meta])))
print('group-cv auc', round(float(np.mean(aucs)),4), '±', round(float(np.std(aucs)),4), aucs)
json.dump({'group_cv_auc_mean':float(np.mean(aucs)),'group_cv_auc_std':float(np.std(aucs)),'n':len(df)},open(OUT/'cv.json','w'),indent=2)
joblib.dump({'pca':pca,'models':models},OUT/'models.joblib')
