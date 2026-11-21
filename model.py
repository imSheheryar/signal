import json, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

OUT=Path('artifacts')
f=pd.read_parquet(OUT/'calls_features.parquet')
dyn=pd.read_parquet(OUT/'call_dynamics.parquet')
lab=pd.read_parquet(OUT/'calls_labels.parquet')
txt=pd.read_parquet(OUT/'calls_text.parquet')
df=f.merge(dyn,on='call_id',how='left').merge(lab.drop(columns=['date','ticker']),on='call_id',how='inner').merge(txt,on='call_id',how='inner')
emb_path=OUT/'text_embeddings.parquet'
emb=pd.read_parquet(emb_path) if emb_path.exists() else None
if emb is not None:
    df=df.merge(emb,on='call_id',how='left')
df=df.sort_values('date').reset_index(drop=True)
print('rows',len(df),'dates',df.date.min().date(),df.date.max().date())

acc=[c for c in df.columns if c.startswith('acc_')]
dyn_c=[c for c in df.columns if c.startswith('dyn_')]
lex=['n_words','n_unique','ttr','n_chars','n_numbers','hedge_rate','neg_rate','fwd_rate','modal_rate','qa_start_frac','has_transition']
sent=[c for c in df.columns if c.startswith('sent_')]
embc=[c for c in df.columns if c.startswith('emb_')]
meta=['n_segments','audio_len']
for c in acc+dyn_c+lex+sent+embc+meta+['pre_vol']:
    df[c]=pd.to_numeric(df[c],errors='coerce')

# operational target: top tercile of post-call volatility, threshold from training period only
train_mask=df.date<'2017-01-01'; val_mask=(df.date>='2017-01-01')&(df.date<'2017-07-01'); test_mask=df.date>='2017-07-01'
thr=df.loc[train_mask,'post_vol'].quantile(2/3)
df['y']=(df.post_vol>=thr).astype(int)
print('threshold',round(thr,4),'rates',round(df.loc[train_mask,'y'].mean(),3),round(df.loc[val_mask,'y'].mean(),3),round(df.loc[test_mask,'y'].mean(),3))

tr=df[train_mask].copy(); va=df[val_mask].copy(); te=df[test_mask].copy()

def fit_lr(X,y):
    return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(max_iter=3000,C=0.5,class_weight='balanced'))]).fit(X,y)
def fit_lgb(X,y):
    return lgb.LGBMClassifier(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=40,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,n_jobs=4,verbose=-1).fit(X,y)
def prob(m,X): return m.predict_proba(X)[:,1]
def met(y,p):
    return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,p)))
def boot(y,p,n=1000):
    rng=np.random.default_rng(42); a=[]; b=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        a.append(roc_auc_score(y[i],p[i])); b.append(average_precision_score(y[i],p[i]))
    return {'auc':(float(np.percentile(a,2.5)),float(np.percentile(a,97.5))),
            'ap':(float(np.percentile(b,2.5)),float(np.percentile(b,97.5)))}

# text representations fit on training only
vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=3,stop_words='english',sublinear_tf=True)
Xtr_txt=vec.fit_transform(tr.text); Xva_txt=vec.transform(va.text); Xte_txt=vec.transform(te.text)
svd=TruncatedSVD(n_components=100,random_state=42).fit(Xtr_txt)
S_tr=svd.transform(Xtr_txt); S_va=svd.transform(Xva_txt); S_te=svd.transform(Xte_txt)
svd_cols=[f'svd_{i}' for i in range(S_tr.shape[1])]
for part,S in [(tr,S_tr),(va,S_va),(te,S_te)]:
    for i,c in enumerate(svd_cols): part[c]=S[:,i]

models={}
models['pre_vol']=fit_lr(tr[['pre_vol']],tr.y)
models['text_tfidf']=LogisticRegression(max_iter=3000,C=0.5,class_weight='balanced').fit(Xtr_txt,tr.y)
models['acoustic']=fit_lgb(tr[acc+dyn_c+meta],tr.y)
fusion_cols=['pre_vol']+acc+dyn_c+lex+sent+meta+svd_cols+embc
models['fusion']=fit_lgb(tr[fusion_cols],tr.y)
# ablations: FinBERT embeddings on their own, and fusion without the 768-d vectors
emb_only=sent+embc
models['text_emb']=fit_lr(tr[emb_only],tr.y)
fusion_no_emb=[c for c in fusion_cols if not c.startswith('emb_')]
models['fusion_no_emb']=fit_lgb(tr[fusion_no_emb],tr.y)
rows=[]; preds={}
def add(name,pv,pt):
    iso=IsotonicRegression(out_of_bounds='clip').fit(pv,va.y)
    # isotonic can output exact 0/1; clipping keeps log loss finite and avoids
    # displaying a 0% risk for a call that still has a non-zero base rate
    pt_cal=np.clip(iso.predict(pt),0.01,0.99)
    m=met(te.y,pt_cal); ci=boot(te.y.values,pt)
    rows.append(dict(model=name,val_auc=float(roc_auc_score(va.y,pv)),**m,
                     auc_lo=ci['auc'][0],auc_hi=ci['auc'][1],ap_lo=ci['ap'][0],ap_hi=ci['ap'][1]))
    preds[name]=pt_cal

add('pre_vol',prob(models['pre_vol'],va[['pre_vol']]),prob(models['pre_vol'],te[['pre_vol']]))
add('text_tfidf',prob(models['text_tfidf'],Xva_txt),prob(models['text_tfidf'],Xte_txt))
add('acoustic',prob(models['acoustic'],va[acc+dyn_c+meta]),prob(models['acoustic'],te[acc+dyn_c+meta]))
add('fusion',prob(models['fusion'],va[fusion_cols]),prob(models['fusion'],te[fusion_cols]))
add('text_emb',prob(models['text_emb'],va[emb_only]),prob(models['text_emb'],te[emb_only]))
add('fusion_no_emb',prob(models['fusion_no_emb'],va[fusion_no_emb]),prob(models['fusion_no_emb'],te[fusion_no_emb]))

# late fusion variants
pv_va=prob(models['pre_vol'],va[['pre_vol']]); pv_te=prob(models['pre_vol'],te[['pre_vol']])
pt_va=prob(models['text_tfidf'],Xva_txt); pt_te=prob(models['text_tfidf'],Xte_txt)
pa_va=prob(models['acoustic'],va[acc+dyn_c+meta]); pa_te=prob(models['acoustic'],te[acc+dyn_c+meta])
pf_va=prob(models['fusion'],va[fusion_cols]); pf_te=prob(models['fusion'],te[fusion_cols])
variants={'late_pre_text':(['pv','pt'],[pv_va,pt_va],[pv_te,pt_te]),
          'late_pre_text_ac':(['pv','pt','pa'],[pv_va,pt_va,pa_va],[pv_te,pt_te,pa_te]),
          'late_all':(['pv','pt','pa','pf'],[pv_va,pt_va,pa_va,pf_va],[pv_te,pt_te,pa_te,pf_te])}
stack=None
for name,(cols,va_p,te_p) in variants.items():
    st=LogisticRegression(max_iter=1000,class_weight='balanced').fit(np.column_stack(va_p),va.y)
    add(name,st.predict_proba(np.column_stack(va_p))[:,1],st.predict_proba(np.column_stack(te_p))[:,1])
    if name=='late_pre_text_ac': stack=st

res=pd.DataFrame(rows).sort_values('auc',ascending=False)
print(res.round(4).to_string(index=False))
res.to_csv(OUT/'metrics.csv',index=False)
preds_out={f'pred_{k}':v for k,v in preds.items()}
pd.DataFrame({'call_id':te.call_id.values,'ticker':te.ticker.values,'date':te.date.values,'y':te.y.values,
              'post_vol':te.post_vol.values,'pre_vol':te.pre_vol.values,**preds_out}).to_parquet(OUT/'test_predictions.parquet',index=False)

# feature importance and SHAP contributions for the fusion model
fi=pd.DataFrame({'feature':fusion_cols,'gain':models['fusion'].booster_.feature_importance('gain')}).sort_values('gain',ascending=False)
fi.to_csv(OUT/'feature_importance.csv',index=False)
print(fi.head(20).to_string(index=False))
contrib=models['fusion'].booster_.predict(te[fusion_cols],pred_contrib=True)
pd.DataFrame(contrib[:,:-1],columns=fusion_cols).assign(call_id=te.call_id.values).to_parquet(OUT/'shap_test.parquet',index=False)

# company-disjoint CV on the fusion model
g=GroupKFold(n_splits=5); aucs=[]
cvdf=pd.concat([tr,va,te],ignore_index=True)
for a,b in g.split(cvdf,cvdf.y,groups=cvdf.ticker):
    m=fit_lgb(cvdf.iloc[a][fusion_cols],cvdf.iloc[a].y)
    aucs.append(roc_auc_score(cvdf.iloc[b].y,prob(m,cvdf.iloc[b][fusion_cols])))
print('group cv',round(float(np.mean(aucs)),4),round(float(np.std(aucs)),4),aucs)
json.dump({'threshold':float(thr),'group_cv_auc_mean':float(np.mean(aucs)),'group_cv_auc_std':float(np.std(aucs)),
           'n_train':len(tr),'n_val':len(va),'n_test':len(te),'embeddings':emb is not None},open(OUT/'model_meta.json','w'),indent=2)
joblib.dump({'vectorizer':vec,'svd':svd,'models':models,'stack':stack,'fusion_cols':fusion_cols,'threshold':thr},OUT/'models.joblib')
print('saved artifacts')
