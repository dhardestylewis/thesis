"""Quick calibration summary for chat output."""
import csv, sys, numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss

csv.field_size_limit(2**31 - 1)

rows_by_year = {}
all_rows = []
with open('Analysis/Results/Diffusion_v3/per_parcel_scores.csv') as f:
    for row in csv.DictReader(f):
        r = {'yr': int(row['year']), 'lr': float(row['lr_score']), 
             'diff': float(row['diff_score']), 'ens': float(row['ensemble_score']),
             'actual': int(row['actual'])}
        all_rows.append(r)
        rows_by_year.setdefault(r['yr'], []).append(r)

def p2l(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def l2p(l):
    return 1.0 / (1.0 + np.exp(-l))

def fit_T(s, y):
    lo = p2l(s)
    res = minimize_scalar(lambda T: log_loss(y, l2p(lo / T)), bounds=(0.1, 10), method='bounded')
    return res.x

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for mk, mn in [('lr', 'LogReg'), ('diff', 'Diffusion'), ('ens', 'Ensemble')]:
    print(f'\n{"="*65}')
    print(f'  {mn}')
    print(f'{"="*65}')
    
    for yr in sorted(rows_by_year):
        rr = rows_by_year[yr]
        s = np.array([r[mk] for r in rr])
        y = np.array([r['actual'] for r in rr])
        
        # Fit T on other years
        os_ = np.array([r[mk] for r in all_rows if r['yr'] != yr])
        oy_ = np.array([r['actual'] for r in all_rows if r['yr'] != yr])
        T = fit_T(os_, oy_)
        cal = l2p(p2l(s) / T)
        
        b0 = brier_score_loss(y, s)
        b1 = brier_score_loss(y, cal)
        n_pos = int(y.sum())
        n_tot = len(y)
        
        print(f'\n  {yr}  T={T:.3f}  Brier: {b0:.4f} -> {b1:.4f}  positives={n_pos}/{n_tot}')
        print(f'  {"Thresh":>8} | {"Raw count":>10} {"Cal count":>10} | {"Protests":>9} | {"Raw rate":>9} {"Cal rate":>9}')
        print(f'  {"-"*8}-+-{"-"*10}-{"-"*10}-+-{"-"*9}-+-{"-"*9}-{"-"*9}')
        
        for t in thresholds:
            nr = int((s > t).sum())
            nc = int((cal > t).sum())
            pr = int(y[s > t].sum()) if nr > 0 else 0
            pc = int(y[cal > t].sum()) if nc > 0 else 0
            rr_ = pr / nr * 100 if nr > 0 else 0
            rc = pc / nc * 100 if nc > 0 else 0
            label = f'>{int(t*100)}%'
            print(f'  {label:>8} | {nr:>10,} {nc:>10,} | {pc:>9} | {rr_:>8.1f}% {rc:>8.1f}%')
