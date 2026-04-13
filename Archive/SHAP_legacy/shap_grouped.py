"""
shap_grouped.py — Grouped SHAP Attribution for Defensible Thesis Claims
========================================================================
Computes per-feature SHAP values, then aggregates them into theoretically
meaningful coalitions to produce defensible group-level Shapley values.

This addresses the multicollinearity/effect-entanglement problem: individual
feature attributions within a correlated group are unreliable, but the group
total is stable and interpretable.

Coalition structure:
  COARSE (7 groups):
    Case Tract Demo | Prop Tract Demo | Δ Demo | Macro | Property | Zoning | Land Use | Geography

  FINE (17 sub-groups):
    Case:Race | Case:Income | Case:Tenure | Case:General
    Prop:Race | Prop:Income | Prop:Tenure | Prop:General
    Δ:Race | Δ:Income | Δ:Tenure | Δ:General
    Macro | Property | Zoning | Land Use | Geography

  SAFE INDIVIDUALS (6 features):
    deed_acreage | delta_acs_renter_occupied_units | macro_mortgage30
    ldb_units | ldb_basezone | lui_general_land_use_100

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
import warnings
import os
import time

warnings.filterwarnings('ignore')

EPOCHS = 120; BATCH_SIZE = 512; LR = 1e-3; VREX_PEN = 100.0; LATENT = 16
MIN_ENV = 5; N_SAMPLE = 33000

PROJECT = r"c:\Users\dhl\data\thesis\thesis"
PANEL = os.path.join(PROJECT, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
CENSUS = os.path.join(PROJECT, "Data", "Panel", "Intermediate", "census_tract_timeseries.csv")
ENVS = os.path.join(PROJECT, "Analysis", "Results", "irm_environment_assignments.csv")

ACS = ['total_population','median_age','race_white','race_black','race_asian',
       'race_hispanic','median_household_income','poverty_count','median_home_value',
       'owner_occupied_units','renter_occupied_units','median_gross_rent','total_housing_units']

FRED = {
    'macro_mortgage30': {2007:6.34,2008:6.03,2009:5.04,2010:4.69,2011:4.45,2012:3.66,2013:3.98,2014:4.17,2015:3.85,2016:3.65,2017:3.99,2018:4.54,2019:3.94,2020:3.11,2021:2.96,2022:5.34,2023:6.81,2024:6.72},
    'macro_fedfunds': {2007:5.02,2008:1.92,2009:0.16,2010:0.18,2011:0.10,2012:0.14,2013:0.11,2014:0.09,2015:0.13,2016:0.39,2017:1.00,2018:1.83,2019:2.16,2020:0.36,2021:0.08,2022:1.68,2023:5.33,2024:5.33},
    'macro_cpi': {2007:207.3,2008:215.3,2009:214.5,2010:218.1,2011:224.9,2012:229.6,2013:233.0,2014:236.7,2015:237.0,2016:240.0,2017:245.1,2018:251.1,2019:255.7,2020:258.8,2021:271.0,2022:292.7,2023:304.7,2024:313.0},
    'macro_unemployment': {2007:4.6,2008:5.8,2009:9.3,2010:9.6,2011:8.9,2012:8.1,2013:7.4,2014:6.2,2015:5.3,2016:4.9,2017:4.4,2018:3.9,2019:3.7,2020:8.1,2021:5.4,2022:3.6,2023:3.6,2024:4.0},
    'macro_housing_starts': {2007:1355,2008:906,2009:554,2010:587,2011:609,2012:781,2013:925,2014:1003,2015:1112,2016:1174,2017:1203,2018:1250,2019:1290,2020:1380,2021:1601,2022:1554,2023:1420,2024:1350},
}

PROP_NUM = ['total_market_value','deed_acreage','land_market_value']
ZONE_NUM = ['ldb_far','ldb_units']
CAT_FEATS = ['property_category_code','council_district','lui_general_land_use','ldb_basezone']


def classify_feature(name):
    """Assign each feature to (coarse_group, fine_group)."""
    # Numeric features
    if name in PROP_NUM:
        return ('Property', 'Property')
    if name in ZONE_NUM:
        return ('Zoning/Density', 'Zoning/Density')

    # ACS features
    for prefix, coarse in [('case_acs_', 'Case Tract'), ('prop_acs_', 'Prop Tract'), ('delta_acs_', 'Δ (Gap)')]:
        if name.startswith(prefix):
            base = name[len(prefix):]
            if 'race' in base:
                return (coarse, f'{coarse}: Race')
            elif 'income' in base or 'poverty' in base:
                return (coarse, f'{coarse}: Income')
            elif 'owner' in base or 'renter' in base or 'rent' in base or 'housing' in base:
                return (coarse, f'{coarse}: Housing/Tenure')
            else:
                return (coarse, f'{coarse}: General')

    # Macros
    if name.startswith('macro_'):
        return ('Macro/FRED', 'Macro/FRED')

    # Categorical OHE
    if name.startswith('property_category_code') or name.startswith('lui_general_land_use'):
        return ('Land Use', 'Land Use')
    if name.startswith('council_district'):
        return ('Geography', 'Geography')
    if name.startswith('ldb_basezone'):
        return ('Zoning/Density', 'Zoning/Density')

    return ('Other', 'Other')


class CVAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d+1,256),nn.SiLU(),nn.Dropout(0.1),nn.Linear(256,128),nn.SiLU())
        self.mu = nn.Linear(128,LATENT); self.lv = nn.Linear(128,LATENT)
        self.dec = nn.Sequential(nn.Linear(LATENT+1,128),nn.SiLU(),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,d))
        self.cls = nn.Sequential(nn.Linear(LATENT,64),nn.SiLU(),nn.Linear(64,1))
    def encode(self,x,y):
        h=self.enc(torch.cat([x,y.view(-1,1)],1)); return self.mu(h),self.lv(h)
    def forward(self,x,y):
        m,lv=self.encode(x,y); z=m+torch.randn_like(m)*torch.exp(.5*lv)
        return self.dec(torch.cat([z,y.view(-1,1)],1)),m,lv,self.cls(m).squeeze(-1)

class Wrap(nn.Module):
    def __init__(self,cvae,br):
        super().__init__(); self.cvae=cvae; self.br=br
    def forward(self,x):
        m,_=self.cvae.encode(x,torch.full((x.shape[0],),self.br)); return self.cvae.cls(m)


def load_data():
    print("Loading data...")
    cols = (['standardized_tcad_id','year','protest','nearby_GEOID','zoning_case_GEOID']
            + PROP_NUM + ZONE_NUM + CAT_FEATS + [f'acs_{v}' for v in ACS])
    panel = pd.read_csv(PANEL, usecols=list(dict.fromkeys(cols)), low_memory=False)
    panel = panel[panel['year'] <= 2024]

    census = pd.read_csv(CENSUS)
    census['geoid'] = census['geoid'].astype(str).str.strip()
    census['vintage'] = census['vintage'].astype(int)

    panel = panel.rename(columns={f'acs_{v}': f'case_acs_{v}' for v in ACS})
    panel['nearby_GEOID'] = panel['nearby_GEOID'].astype(str).str.strip()
    panel['nearby_tract'] = panel['nearby_GEOID'].apply(lambda x: x[:11] if len(str(x))>=11 else '')
    panel['avm'] = panel['year'].apply(lambda yr: max([v for v in range(2009,2024) if v<=yr], default=2009))

    cp = census.rename(columns={'geoid':'nearby_tract','vintage':'avm',**{v:f'prop_acs_{v}' for v in ACS}})
    panel = panel.merge(cp[['nearby_tract','avm']+[f'prop_acs_{v}' for v in ACS]], on=['nearby_tract','avm'], how='left')

    for v in ACS:
        panel[f'case_acs_{v}'] = pd.to_numeric(panel[f'case_acs_{v}'], errors='coerce')
        panel[f'prop_acs_{v}'] = pd.to_numeric(panel[f'prop_acs_{v}'], errors='coerce')
        panel[f'delta_acs_{v}'] = panel[f'prop_acs_{v}'] - panel[f'case_acs_{v}']

    for k,v in FRED.items(): panel[k] = panel['year'].map(v)

    env = pd.read_csv(ENVS).rename(columns={'CASE_NUMBER':'env_id'})
    panel['standardized_tcad_id'] = panel['standardized_tcad_id'].astype(str)
    env['standardized_tcad_id'] = env['standardized_tcad_id'].astype(str)
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BG')
    es = df.groupby('env_id').size(); df = df[df['env_id'].isin(es[es>=MIN_ENV].index)]
    em = {n:i for i,n in enumerate(df['env_id'].unique())}; df['el'] = df['env_id'].map(em)

    pos = df[df['protest']==1]; neg = df[df['protest']==0].sample(n=N_SAMPLE-len(pos), random_state=42)
    df = pd.concat([pos,neg]).sample(frac=1,random_state=42).reset_index(drop=True)

    all_num = PROP_NUM + ZONE_NUM + [f'case_acs_{v}' for v in ACS] + [f'prop_acs_{v}' for v in ACS] + [f'delta_acs_{v}' for v in ACS] + list(FRED.keys())
    for c in all_num: df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    for c in CAT_FEATS: df[c]=df[c].fillna('Missing').astype(str)

    sc = StandardScaler(); ohe = OneHotEncoder(sparse_output=False,handle_unknown='ignore',max_categories=15)
    Xn = sc.fit_transform(df[all_num]); Xc = ohe.fit_transform(df[CAT_FEATS])
    X = np.hstack([Xn,Xc]).astype(np.float32)
    y = df['protest'].values.astype(np.float32)
    envs = df['el'].values.astype(np.int64)
    fnames = all_num + list(ohe.get_feature_names_out(CAT_FEATS))
    return X, y, envs, fnames


def train(X, y, envs, method):
    model = CVAE(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X),torch.FloatTensor(y),torch.LongTensor(envs)),
                        batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for ep in range(EPOCHS):
        for xb,yb,eb in loader:
            opt.zero_grad()
            rec,m,lv,cl = model(xb,yb)
            loss_per = torch.sum((rec-xb)**2,1) - 0.5*torch.sum(1+lv-m**2-lv.exp(),1) + \
                       nn.functional.binary_cross_entropy_with_logits(cl,yb,reduction='none')
            ue = torch.unique(eb)
            er = [loss_per[eb==e].mean() for e in ue if (eb==e).sum()>=2]
            if len(er)<2: continue
            s = torch.stack(er); erm = s.mean()
            if method=="V-REx":
                b = VREX_PEN if ep>20 else VREX_PEN*(ep/20.0)
                loss = erm + b*s.var()
            else: loss = erm
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if (ep+1)%30==0: print(f"  Epoch {ep+1:3d}")
    return model


def main():
    t0 = time.time()
    X, y, envs, fnames = load_data()

    # Build group mappings
    coarse_map = {}; fine_map = {}
    for fn in fnames:
        c, f = classify_feature(fn)
        coarse_map[fn] = c; fine_map[fn] = f

    for method in ["ERM", "V-REx"]:
        seed = 42 if method=="ERM" else 123
        print(f"\n{'='*70}")
        print(f"Training CVAE ({method})...")
        print(f"{'='*70}")
        torch.manual_seed(seed); np.random.seed(seed)
        model = train(X, y, envs, method)

        # SHAP
        wrap = Wrap(model, y.mean()); wrap.eval()
        Xt = torch.FloatTensor(X)
        np.random.seed(42)
        bg = np.random.choice(len(X),200,replace=False)
        pi = np.where(y==1)[0]; ni = np.where(y==0)[0]
        ei = np.concatenate([np.random.choice(pi,min(500,len(pi)),replace=False),
                             np.random.choice(ni,500,replace=False)])

        print(f"  Running SHAP...")
        exp = shap.GradientExplainer(wrap, Xt[bg])
        sv = exp.shap_values(Xt[ei])
        if isinstance(sv,list): sv=sv[0]
        if sv.ndim==3: sv=sv[:,:,0]

        mean_abs = np.mean(np.abs(sv), axis=0)
        total = mean_abs.sum()

        # ── COARSE GROUP ATTRIBUTION ──
        print(f"\n{'='*70}")
        print(f"COARSE GROUP ATTRIBUTION — {method}")
        print(f"{'='*70}")
        cg = {}
        for i,fn in enumerate(fnames):
            g = coarse_map[fn]; cg[g] = cg.get(g,0) + mean_abs[i]
        print(f"{'Group':<25} {'|SHAP|':>12} {'%':>7}  Bar")
        print("-"*60)
        for g,v in sorted(cg.items(), key=lambda x:-x[1]):
            p = 100*v/total
            print(f"{g:<25} {v:>12.6f} {p:>6.1f}%  {'█'*int(p/2)}")

        # ── FINE SUB-GROUP ATTRIBUTION ──
        print(f"\n{'='*70}")
        print(f"FINE SUB-GROUP ATTRIBUTION — {method}")
        print(f"{'='*70}")
        fg = {}
        for i,fn in enumerate(fnames):
            g = fine_map[fn]; fg[g] = fg.get(g,0) + mean_abs[i]
        print(f"{'Sub-Group':<30} {'|SHAP|':>12} {'%':>7}  Bar")
        print("-"*65)
        for g,v in sorted(fg.items(), key=lambda x:-x[1]):
            p = 100*v/total
            print(f"{g:<30} {v:>12.6f} {p:>6.1f}%  {'█'*int(p)}")

        # ── SAFE INDIVIDUAL FEATURES ──
        safe_features = ['deed_acreage', 'delta_acs_renter_occupied_units',
                         'macro_mortgage30', 'ldb_units']
        # Also include any OHE with basezone or lui_100
        for fn in fnames:
            if 'ldb_basezone' in fn or fn == 'lui_general_land_use_100':
                safe_features.append(fn)

        print(f"\n{'='*70}")
        print(f"SAFE INDIVIDUAL FEATURES — {method}")
        print(f"{'='*70}")
        print("(These features have |φ-corr| < 0.5 with all other top features)")
        print(f"{'Feature':<50} {'|SHAP|':>10} {'%':>7}  {'Direction':<15}")
        print("-"*85)
        for fn in safe_features:
            if fn in fnames:
                idx = fnames.index(fn)
                p = 100*mean_abs[idx]/total
                if p < 0.2: continue
                d = "↑ INCREASES" if np.mean(sv[:,idx])>0 else "↓ DECREASES"
                print(f"{fn:<50} {mean_abs[idx]:>10.6f} {p:>6.1f}%  {d} protest")

        # ── ERM vs V-REx DELTA (for comparison) ──
        if method == "ERM":
            erm_coarse = dict(cg)
            erm_fine = dict(fg)
        else:
            print(f"\n{'='*70}")
            print(f"ERM → V-REx SHIFT (Coarse Groups)")
            print(f"{'='*70}")
            print(f"{'Group':<25} {'ERM %':>8} {'V-REx %':>8} {'Shift':>10}")
            print("-"*55)
            for g in sorted(set(list(erm_coarse.keys()) + list(cg.keys()))):
                e_pct = 100*erm_coarse.get(g,0)/total if erm_coarse else 0
                v_pct = 100*cg.get(g,0)/total
                shift = v_pct - e_pct
                arrow = "↑" if shift > 0 else "↓"
                print(f"{g:<25} {e_pct:>7.1f}% {v_pct:>7.1f}% {arrow} {abs(shift):>6.1f}pp")

    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
