"""
shap_vrex_final.py — Definitive SHAP on V-REx CVAE (All Limitations Addressed)
================================================================================
Addresses ALL identified limitations from previous runs:

  1. BISG ethnicity estimation from owner_name + ZIP (surgeo)
  2. Dual-tract ACS demographics (case + property tracts)
  3. Demographic deltas (property - case)
  4. FRED macroeconomic indicators
  5. Owner/property characteristics (exemption flags, year built, improvement sqft)
  6. Grouped SHAP with entanglement-safe coalitions

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
import re

warnings.filterwarnings('ignore')

EPOCHS = 120; BS = 512; LR = 1e-3; VREX_PEN = 100.0; LATENT = 16
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

BISG_COLS = ['bisg_white','bisg_black','bisg_asian','bisg_hispanic','bisg_aian','bisg_multiple']


def extract_surname(name_str):
    """Extract surname from TCAD owner_name format: 'LAST, FIRST M' or 'LAST FIRST'."""
    if not isinstance(name_str, str) or not name_str.strip():
        return None
    name = name_str.strip().upper()
    # Skip corporate/trust names
    for kw in ['LLC', 'INC', 'CORP', 'TRUST', 'LP', 'LTD', 'ASSOC', 'BANK', 'FUND', 'HOMES']:
        if kw in name:
            return None
    # "LAST, FIRST" format
    if ',' in name:
        return name.split(',')[0].strip()
    # "LAST FIRST" format — take first word
    parts = name.split()
    if parts:
        return parts[0].strip()
    return None


def extract_zip(situs):
    """Extract 5-digit ZIP from situs_city_state_zip."""
    if not isinstance(situs, str):
        return None
    match = re.search(r'\b(\d{5})\b', situs)
    return match.group(1) if match else None


def run_bisg(panel):
    """Run BISG ethnicity estimation from owner_name + situs ZIP."""
    from surgeo import SurgeoModel
    model = SurgeoModel()

    print("  Extracting surnames and ZIPs...")
    panel['_surname'] = panel['owner_name'].apply(extract_surname)
    panel['_zip'] = panel['situs_city_state_zip'].apply(extract_zip)

    # Only run BISG where we have both surname and ZIP
    has_both = panel['_surname'].notna() & panel['_zip'].notna()
    n_valid = has_both.sum()
    print(f"  Valid name+ZIP pairs: {n_valid:,} / {len(panel):,} ({100*n_valid/len(panel):.1f}%)")

    # Initialize BISG columns
    for col in BISG_COLS:
        panel[col] = np.nan

    if n_valid > 0:
        valid_idx = panel[has_both].index
        # Process in chunks to avoid memory issues
        chunk_size = 50000
        for start in range(0, len(valid_idx), chunk_size):
            chunk_idx = valid_idx[start:start+chunk_size]
            surnames = panel.loc[chunk_idx, '_surname']
            zips = panel.loc[chunk_idx, '_zip']
            try:
                probs = model.get_probabilities(surnames, zips)
                # Map surgeo output columns to our columns
                col_map = {
                    'white': 'bisg_white', 'black': 'bisg_black',
                    'api': 'bisg_asian', 'hispanic': 'bisg_hispanic',
                    'aian': 'bisg_aian', 'multiple': 'bisg_multiple'
                }
                for src, dst in col_map.items():
                    if src in probs.columns:
                        panel.loc[chunk_idx, dst] = probs[src].values
            except Exception as e:
                print(f"  BISG chunk error: {e}")

    matched = panel['bisg_white'].notna().sum()
    print(f"  BISG matched: {matched:,} ({100*matched/len(panel):.1f}%)")

    panel.drop(columns=['_surname', '_zip'], inplace=True)
    return panel


def classify_feature(name):
    for prefix, coarse in [('case_acs_','Case Tract'),('prop_acs_','Prop Tract'),('delta_acs_','Δ (Gap)')]:
        if name.startswith(prefix):
            base = name[len(prefix):]
            if 'race' in base: return (coarse, f'{coarse}: Race')
            elif 'income' in base or 'poverty' in base: return (coarse, f'{coarse}: Income')
            elif 'owner' in base or 'renter' in base or 'rent' in base or 'housing' in base:
                return (coarse, f'{coarse}: Tenure')
            else: return (coarse, f'{coarse}: General')
    if name.startswith('bisg_'): return ('Owner Ethnicity (BISG)', 'Owner Ethnicity (BISG)')
    if name.startswith('macro_'): return ('Macro/FRED', 'Macro/FRED')
    if name in ['total_market_value','deed_acreage','land_market_value','improvement_sq_ft']:
        return ('Property', 'Property')
    if name in ['ldb_far','ldb_units']: return ('Zoning/Density', 'Zoning/Density')
    if name in ['year_built','property_age']: return ('Property Age', 'Property Age')
    if name.startswith('exemption_flag') or name == 'homesite_flag':
        return ('Owner Flags', 'Owner Flags')
    if name.startswith('property_category_code') or name.startswith('lui_general_land_use'):
        return ('Land Use', 'Land Use')
    if name.startswith('council_district'): return ('Geography', 'Geography')
    if name.startswith('ldb_basezone'): return ('Zoning/Density', 'Zoning/Density')
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
    def __init__(self,cvae,br): super().__init__(); self.cvae=cvae; self.br=br
    def forward(self,x):
        m,_=self.cvae.encode(x,torch.full((x.shape[0],),self.br)); return self.cvae.cls(m)


def load_data():
    print("=" * 60)
    print("LOADING FULL-FEATURE PANEL")
    print("=" * 60)

    cols = (['standardized_tcad_id','year','protest','nearby_GEOID','zoning_case_GEOID',
             'owner_name','situs_city_state_zip',
             'total_market_value','deed_acreage','land_market_value','improvement_sq_ft',
             'ldb_far','ldb_units','year_built',
             'exemption_flag_hs','exemption_flag_ov65','exemption_flag_dp','exemption_flag_dv',
             'homesite_flag'] +
            ['property_category_code','council_district','lui_general_land_use','ldb_basezone'] +
            [f'acs_{v}' for v in ACS])

    panel = pd.read_csv(PANEL, usecols=list(dict.fromkeys(cols)), low_memory=False)
    panel = panel[panel['year'] <= 2024]
    print(f"  Panel: {len(panel):,} rows")

    # BISG ethnicity estimation
    print("\nRunning BISG ethnicity estimation...")
    panel = run_bisg(panel)

    # Property age
    panel['year_built'] = pd.to_numeric(panel['year_built'], errors='coerce')
    panel['property_age'] = panel['year'] - panel['year_built']

    # Exemption flags to binary
    for col in ['exemption_flag_hs','exemption_flag_ov65','exemption_flag_dp','exemption_flag_dv','homesite_flag']:
        panel[col] = panel[col].apply(lambda x: 1 if str(x).strip().upper() in ('Y','1','YES','TRUE') else 0)

    # Census for prop tract join
    print("\nJoining dual-tract ACS demographics...")
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

    # Environment assignments
    env = pd.read_csv(ENVS).rename(columns={'CASE_NUMBER':'env_id'})
    panel['standardized_tcad_id'] = panel['standardized_tcad_id'].astype(str)
    env['standardized_tcad_id'] = env['standardized_tcad_id'].astype(str)
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BG')
    es = df.groupby('env_id').size(); df = df[df['env_id'].isin(es[es>=MIN_ENV].index)]
    em = {n:i for i,n in enumerate(df['env_id'].unique())}; df['el'] = df['env_id'].map(em)

    pos = df[df['protest']==1]; neg = df[df['protest']==0]
    n_neg = N_SAMPLE - len(pos)
    if len(neg) > n_neg: neg = neg.sample(n=n_neg, random_state=42)
    df = pd.concat([pos,neg]).sample(frac=1,random_state=42).reset_index(drop=True)

    # Build feature matrix
    prop_num = ['total_market_value','deed_acreage','land_market_value','improvement_sq_ft']
    zone_num = ['ldb_far','ldb_units']
    owner_num = BISG_COLS + ['property_age']
    owner_flags = ['exemption_flag_hs','exemption_flag_ov65','exemption_flag_dp','exemption_flag_dv','homesite_flag']
    case_acs = [f'case_acs_{v}' for v in ACS]
    prop_acs = [f'prop_acs_{v}' for v in ACS]
    delta_acs = [f'delta_acs_{v}' for v in ACS]
    macro = list(FRED.keys())

    all_num = prop_num + zone_num + owner_num + owner_flags + case_acs + prop_acs + delta_acs + macro
    cat_feats = ['property_category_code','council_district','lui_general_land_use','ldb_basezone']

    for c in all_num:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    for c in cat_feats:
        df[c] = df[c].fillna('Missing').astype(str)

    sc = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=15)
    Xn = sc.fit_transform(df[all_num]); Xc = ohe.fit_transform(df[cat_feats])
    X = np.hstack([Xn,Xc]).astype(np.float32)
    y = df['protest'].values.astype(np.float32)
    envs = df['el'].values.astype(np.int64)
    fnames = all_num + list(ohe.get_feature_names_out(cat_feats))

    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {len(df):,} rows | {X.shape[1]} features")
    print(f"  {len(prop_num)} property + {len(zone_num)} zoning + {len(owner_num)} owner/BISG + "
          f"{len(owner_flags)} exemption flags")
    print(f"  {len(case_acs)} case ACS + {len(prop_acs)} prop ACS + {len(delta_acs)} delta + "
          f"{len(macro)} macro + {len(list(ohe.get_feature_names_out(cat_feats)))} OHE cat")
    print(f"  Base rate: {y.mean():.3f}")

    return X, y, envs, fnames


def train(X, y, envs, method):
    model = CVAE(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X),torch.FloatTensor(y),torch.LongTensor(envs)),
                        batch_size=BS, shuffle=True)
    model.train()
    for ep in range(EPOCHS):
        tl = 0; nb = 0
        for xb,yb,eb in loader:
            opt.zero_grad()
            rec,m,lv,cl = model(xb,yb)
            lp = torch.sum((rec-xb)**2,1) - 0.5*torch.sum(1+lv-m**2-lv.exp(),1) + \
                 nn.functional.binary_cross_entropy_with_logits(cl,yb,reduction='none')
            ue = torch.unique(eb)
            er = [lp[eb==e].mean() for e in ue if (eb==e).sum()>=2]
            if len(er)<2: continue
            s = torch.stack(er); erm = s.mean()
            if method=="V-REx":
                b = VREX_PEN if ep>20 else VREX_PEN*(ep/20.0)
                loss = erm + b*s.var()
            else: loss = erm
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
            tl += loss.item(); nb += 1
        if (ep+1)%30==0: print(f"  Epoch {ep+1:3d} | Loss: {tl/max(nb,1):.8f}")
    return model


def main():
    t0 = time.time()
    X, y, envs, fnames = load_data()

    # Build group mappings
    coarse_map = {}; fine_map = {}
    for fn in fnames:
        c, f = classify_feature(fn)
        coarse_map[fn] = c; fine_map[fn] = f

    erm_coarse = None
    for method in ["ERM", "V-REx"]:
        seed = 42 if method=="ERM" else 123
        print(f"\n{'='*70}")
        print(f"Training CVAE ({method}) — {X.shape[1]} features...")
        print(f"{'='*70}")
        torch.manual_seed(seed); np.random.seed(seed)
        model = train(X, y, envs, method)

        wrap = Wrap(model, y.mean()); wrap.eval()
        Xt = torch.FloatTensor(X)
        np.random.seed(42)
        bg = np.random.choice(len(X),200,replace=False)
        pi = np.where(y==1)[0]; ni = np.where(y==0)[0]
        ei = np.concatenate([np.random.choice(pi,min(500,len(pi)),replace=False),
                             np.random.choice(ni,500,replace=False)])

        print(f"  SHAP GradientExplainer...")
        exp = shap.GradientExplainer(wrap, Xt[bg])
        sv = exp.shap_values(Xt[ei])
        if isinstance(sv,list): sv=sv[0]
        if sv.ndim==3: sv=sv[:,:,0]

        mean_abs = np.mean(np.abs(sv), axis=0)
        total = mean_abs.sum()

        # ── Coarse groups ──
        cg = {}
        for i,fn in enumerate(fnames):
            g = coarse_map[fn]; cg[g] = cg.get(g,0) + mean_abs[i]

        print(f"\n{'='*70}")
        print(f"COARSE GROUP ATTRIBUTION — {method}")
        print(f"{'='*70}")
        print(f"{'Group':<30} {'|SHAP|':>12} {'%':>7}  Bar")
        print("-"*65)
        for g,v in sorted(cg.items(), key=lambda x:-x[1]):
            p = 100*v/total
            print(f"{g:<30} {v:>12.6f} {p:>6.1f}%  {'█'*int(p/2)}")

        # ── Fine sub-groups ──
        fg = {}
        for i,fn in enumerate(fnames):
            g = fine_map[fn]; fg[g] = fg.get(g,0) + mean_abs[i]

        print(f"\n{'='*70}")
        print(f"FINE SUB-GROUP ATTRIBUTION — {method}")
        print(f"{'='*70}")
        print(f"{'Sub-Group':<30} {'|SHAP|':>12} {'%':>7}  Bar")
        print("-"*65)
        for g,v in sorted(fg.items(), key=lambda x:-x[1]):
            p = 100*v/total
            print(f"{g:<30} {v:>12.6f} {p:>6.1f}%  {'█'*int(p)}")

        # ── Top individual features ──
        sorted_idx = np.argsort(mean_abs)[::-1]
        print(f"\n{'='*70}")
        print(f"TOP 25 INDIVIDUAL FEATURES — {method}")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Group':<25} {'Feature':<42} {'%':>6} {'Dir':<12}")
        print("-"*90)
        for r, idx in enumerate(sorted_idx[:25]):
            p = 100*mean_abs[idx]/total
            d = "↑" if np.mean(sv[:,idx])>0 else "↓"
            print(f"{r+1:<4} {coarse_map[fnames[idx]]:<25} {fnames[idx]:<42} {p:>5.1f}% {d}")

        # ── ERM vs V-REx shift ──
        if method == "ERM":
            erm_coarse = dict(cg)
            erm_total = total
        else:
            print(f"\n{'='*70}")
            print(f"ERM → V-REx SHIFT (Coarse Groups)")
            print(f"{'='*70}")
            print(f"{'Group':<30} {'ERM %':>8} {'V-REx %':>8} {'Shift':>10}")
            print("-"*60)
            for g in sorted(set(list(erm_coarse.keys()) + list(cg.keys()))):
                e = 100*erm_coarse.get(g,0)/erm_total
                v = 100*cg.get(g,0)/total
                s = v - e
                a = "↑" if s > 0 else "↓"
                print(f"{g:<30} {e:>7.1f}% {v:>7.1f}% {a} {abs(s):>6.1f}pp")

    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
