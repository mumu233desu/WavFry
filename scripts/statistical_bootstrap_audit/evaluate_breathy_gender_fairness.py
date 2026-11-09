import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score

def compute_metrics(yt, yp, threshold=0.5):
    yt = np.array(yt, dtype=int)
    yp = np.array(yp, dtype=float)
    y_pred = (yp >= threshold).astype(int)
    
    acc = float(accuracy_score(yt, y_pred))
    prec = float(precision_score(yt, y_pred, zero_division=0.0))
    rec = float(recall_score(yt, y_pred, zero_division=0.0))
    f1 = float(f1_score(yt, y_pred, zero_division=0.0))
    
    p_c, r_c, _ = precision_recall_curve(yt, yp)
    pr_auc = float(auc(r_c, p_c))
    roc_auc = float(roc_auc_score(yt, yp))
    
    return {
        "N": len(yt),
        "Positive_Rate": float(np.mean(yt)),
        "Pos_Count": int(np.sum(yt)),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc
    }

from pathlib import Path

# Load clean_preds_temp.csv
ROOT = Path(__file__).resolve().parents[2]
preds_file = ROOT / 'data' / 'experiments' / 'interpretability' / 'clean_preds_temp.csv'
if not preds_file.exists():
    preds_file = Path('data/experiments/interpretability/clean_preds_temp.csv')

data = []
with open(preds_file, 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        lb = r['label_breathy']
        if lb != '':
            fk = r['file_key']
            g = 'Female' if fk.startswith('F') else ('Male' if fk.startswith('M') else 'Unknown')
            data.append({
                'gender': g,
                'label': float(lb),
                'prob': float(r['prob_breathy']),
                'speaker_id': r['speaker_id'],
                'dataset_id': r['dataset_id']
            })

df = pd.DataFrame(data)
print(f"Total evaluated breathy vowels: {len(df)}")
print(df['gender'].value_counts())

# Best threshold on overall or 0.29 (the validation/frozen threshold for breathy)
# Let's test both tau = 0.5, tau = 0.43 (creaky frozen), and tau = 0.29 (breathy best threshold from Table 1 / topk)
for tau in [0.29, 0.43, 0.50]:
    print(f"\n=================== Threshold tau = {tau} ===================")
    overall = compute_metrics(df['label'], df['prob'], threshold=tau)
    females = compute_metrics(df[df['gender'] == 'Female']['label'], df[df['gender'] == 'Female']['prob'], threshold=tau)
    males = compute_metrics(df[df['gender'] == 'Male']['label'], df[df['gender'] == 'Male']['prob'], threshold=tau)
    
    print(f"{'Cohort':<10} {'N':<6} {'Base Rate':<10} {'Acc.':<8} {'Prec.':<8} {'Rec.':<8} {'F1':<8} {'PR-AUC':<8} {'ROC-AUC':<8}")
    print(f"{'Overall':<10} {overall['N']:<6} {overall['Positive_Rate']*100:.2f}% ({overall['Pos_Count']}) {overall['Accuracy']:.4f}   {overall['Precision']:.4f}   {overall['Recall']:.4f}   {overall['F1']:.4f}   {overall['PR-AUC']:.4f}   {overall['ROC-AUC']:.4f}")
    print(f"{'Female':<10} {females['N']:<6} {females['Positive_Rate']*100:.2f}% ({females['Pos_Count']}) {females['Accuracy']:.4f}   {females['Precision']:.4f}   {females['Recall']:.4f}   {females['F1']:.4f}   {females['PR-AUC']:.4f}   {females['ROC-AUC']:.4f}")
    print(f"{'Male':<10} {males['N']:<6} {males['Positive_Rate']*100:.2f}% ({males['Pos_Count']}) {males['Accuracy']:.4f}   {males['Precision']:.4f}   {males['Recall']:.4f}   {males['F1']:.4f}   {males['PR-AUC']:.4f}   {males['ROC-AUC']:.4f}")
    
    delta_pr = abs(females['PR-AUC'] - males['PR-AUC'])
    delta_roc = abs(females['ROC-AUC'] - males['ROC-AUC'])
    delta_f1 = abs(females['F1'] - males['F1'])
    delta_acc = abs(females['Accuracy'] - males['Accuracy'])
    print(f"Disparity (|F - M|): Delta PR-AUC = {delta_pr:.4f}, Delta ROC-AUC = {delta_roc:.4f}, Delta F1 = {delta_f1:.4f}, Delta Acc = {delta_acc:.4f}")
