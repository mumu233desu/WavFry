"""Evaluate a frozen creaky/breathy checkpoint on labeled interval rows by groups."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.data.dataset import CreakyVowelDataset, collate_fn

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--csv',required=True); ap.add_argument('--out',default='scratch/breathy_group_predictions.csv'); ap.add_argument('--batch-size',type=int,default=16); a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=ck['config']; cfg['training']['paradigm']='segment'
    d=pd.read_csv(a.csv); d=d[d.label_breathy.notna()].copy().reset_index(drop=True)
    ds=CreakyVowelDataset(d, context_padding_ms=30); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=collate_fn)
    m=WavFryClassifier(cfg); m.load_state_dict(ck['model_state_dict'],strict=False); m.eval(); out=[]
    with torch.no_grad():
      for wav, lab, lengths, spk, meta in dl:
        z=m(wav,lengths=lengths)['breathy']; out.extend(torch.sigmoid(z).numpy().tolist())
    d['pred_breathy']=out; d.to_csv(a.out,index=False)
    y=d.label_breathy.astype(int); p=d.pred_breathy
    print('overall',len(d),'pos',int(y.sum()),'prauc',average_precision_score(y,p),'roc',roc_auc_score(y,p))
    for c in ['dataset_id','language','speaker_id','phone','phone_type','batch_split']:
      if c not in d: continue
      rows=[]
      for k,g in d.groupby(c,dropna=False):
        if g.label_breathy.nunique()<2: continue
        rows.append((k,len(g),int(g.label_breathy.sum()),average_precision_score(g.label_breathy,g.pred_breathy),roc_auc_score(g.label_breathy,g.pred_breathy),float(g.pred_breathy.mean())))
      print('\n'+c); print(pd.DataFrame(rows,columns=[c,'n','pos','prauc','roc','mean_pred']).sort_values('n',ascending=False).head(40).to_string(index=False))
if __name__=='__main__': main()
