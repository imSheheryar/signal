import os, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(min(8, os.cpu_count() or 4))
name='ProsusAI/finbert'
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForSequenceClassification.from_pretrained(name).eval()
id2label=model.config.id2label
print('labels', id2label, flush=True)
texts=pd.read_parquet('artifacts/calls_text.parquet')
rows=[]; t0=time.time(); B=16; MAX=254
for start in range(0,len(texts),B):
    batch=texts.iloc[start:start+B]
    chunks=[]; ns=[]
    for _,r in batch.iterrows():
        ids=tok(r.text, add_special_tokens=False, truncation=False)['input_ids']
        if len(ids)<=MAX:
            parts=[ids]; n=1
        else:
            parts=[ids[:MAX], ids[-MAX:]]; n=2
        chunks.extend(parts); ns.append(n)
    enc=tok.pad({'input_ids':chunks}, padding=True, return_tensors='pt')
    with torch.inference_mode():
        out=model(**enc, output_hidden_states=True)
    h=out.hidden_states[-1]; mask=enc['attention_mask'].unsqueeze(-1)
    emb=((h*mask).sum(1)/mask.sum(1).clamp(min=1)).numpy()
    probs=torch.softmax(out.logits, dim=-1).numpy()
    idx=0
    for (_,r),n in zip(batch.iterrows(),ns):
        e=emb[idx:idx+n].mean(0); p=probs[idx:idx+n].mean(0); idx+=n
        rec={'call_id':r.call_id}
        rec.update({f'emb_{i}':float(v) for i,v in enumerate(e)})
        for j,lab in id2label.items(): rec[f'sent_{lab.lower()}']=float(p[j])
        rows.append(rec)
    if start % 320 == 0:
        print(start, round(time.time()-t0,1),'s', flush=True)
df=pd.DataFrame(rows)
df.to_parquet('artifacts/text_embeddings.parquet', index=False)
print('saved', df.shape, round(time.time()-t0,1),'s', flush=True)
