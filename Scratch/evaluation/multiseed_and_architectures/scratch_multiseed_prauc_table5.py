import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
DRAFT_DIR = os.path.join(ROOT, 'Thesis_Draft')

print("[*] Loading fully lagged V2 production dataset...")
df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

def map_zoning_density(zone_str):
    if pd.isna(zone_str): return 0
    z = str(zone_str).upper()
    if 'CBD' in z: return 10
    if 'MF-6' in z: return 9
    if 'MF-5' in z: return 8
    if 'MF-4' in z or 'MF-3' in z: return 7
    if 'MF' in z: return 6
    if 'CS' in z or 'GR' in z or 'CH' in z: return 5
    if 'SF-6' in z or 'SF-5' in z: return 4
    if 'SF-4' in z: return 3
    if 'SF-3' in z: return 2
    if 'SF-2' in z or 'SF-1' in z: return 1
    if 'RR' in z or 'DR' in z: return 0.5
    return 0

# Apply ordinal binning to zoning codes to prevent string memorization
if 'zoning_code' in df.columns:
    df['zoning_density_score'] = df['zoning_code'].apply(map_zoning_density)

# --- NEW: Prevent Memorization of Continuous Spatial Features ---
# Trees can memorize exact addresses if we give them exact floating-point lat/lon or highly specific lot sizes
if 'latitude' in df.columns and 'longitude' in df.columns:
    # Round coordinates to ~0.01 (~1.1 km precision) to force neighborhood-level spatial blocks
    df['latitude'] = np.round(df['latitude'], 2)
    df['longitude'] = np.round(df['longitude'], 2)

continuous_targets = ['gross_site_area_acres', 'improvement_sq_ft', 'total_market_value', 'appraised_value']
for col in continuous_targets:
    if col in df.columns:
        # Collapse exact numerical values into 10 decile buckets (1-10 ordinal scale)
        try:
            df[col] = pd.qcut(df[col].replace(0, np.nan), q=10, labels=False, duplicates='drop').fillna(0)
        except Exception:
            pass

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['is_protested'].values
years = df['year'].values

anchors = [2018, 2019, 2020, 2021, 2022, 2023]
eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# 5 seeds for live preview 
seeds = [42, 117, 300, 999, 2026]

print(f"[*] Running {len(seeds)} seeds x {len(anchors)} anchors x {len(eval_years)} eval years...")

all_results = []

for seed_idx, seed in enumerate(seeds):
    print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)})...")

    for anchor in anchors:
        train_mask = years < anchor
        if train_mask.sum() < 50: continue
        X_train_raw = X_raw.values[train_mask]
        y_train = y[train_mask]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)

        models = {
            'CatBoost': CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=seed),
            'XGBoost': XGBClassifier(n_estimators=300, max_depth=6, random_state=seed, eval_metric='logloss', verbosity=0),
            'LightGBM': LGBMClassifier(n_estimators=300, max_depth=6, random_state=seed, verbose=-1),
            'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed),
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=seed, max_iter=200),
        }

        for name, m in models.items():
            try:
                if name in ['Logistic Regression']:
                    m.fit(X_train_sc, y_train)
                else:
                    m.fit(X_train_raw, y_train)
            except Exception as e:
                print(f"    WARN: {name} fit failed: {e}")
                continue

        for test_year in eval_years:
            if test_year < anchor: continue
            test_mask = years == test_year
            if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue

            X_test_raw = X_raw.values[test_mask]
            X_test_sc = scaler.transform(X_test_raw)
            y_test = y[test_mask]
            base_rate = y_test.mean()

            for name, m in models.items():
                try:
                    p = m.predict_proba(X_test_sc if name in ['Logistic Regression'] else X_test_raw)[:, 1]
                    prauc = average_precision_score(y_test, p)
                    lift = prauc / base_rate if base_rate > 0 else 0.0
                    all_results.append({
                        'Seed': seed,
                        'Model': name,
                        'Anchor': anchor,
                        'Evaluate_Year': test_year,
                        'Offset': test_year - anchor,
                        'PRAUC': prauc,
                        'Base_Rate': base_rate,
                        'Lift': lift
                    })
                except Exception as e:
                    print(f"    WARN: {name} predict failed: {e}")

print("[*] Aggregating and saving results...")
res_df = pd.DataFrame(all_results)
out_csv = os.path.join(DRAFT_DIR, 'Multiseed_PRAUC_Table5_Validation.csv')
res_df.to_csv(out_csv, index=False)

# ---- Table 5 style: mean ± std PR-AUC lift, by anchor x eval year ----
print("\n=== Multiseed Table 5: Mean PRAUC Lift (averaged across 20 seeds) ===")
for model in ['CatBoost', 'LightGBM', 'XGBoost', 'Random Forest', 'Logistic Regression']:
    sub = res_df[res_df['Model'] == model]
    pivot = sub.groupby(['Anchor', 'Evaluate_Year'])['Lift'].agg(['mean','std']).reset_index()
    pivot_tbl = pivot.pivot_table(index='Anchor', columns='Evaluate_Year', values='mean')
    std_tbl   = pivot.pivot_table(index='Anchor', columns='Evaluate_Year', values='std')
    print(f"\n{model} — PR-AUC Lift (mean ± std across {len(seeds)} seeds):")
    print(pivot_tbl.round(2).to_string())
    print("Std devs:")
    print(std_tbl.round(3).to_string())

# ---- Summary: headline OOD PR-AUC (2022 holdout = most comparable to thesis OOD) ----
print("\n=== Headline OOD Summary (Test Year 2023-2024, all anchors <= 2022) ===")
ood = res_df[(res_df['Evaluate_Year'] >= 2023) & (res_df['Anchor'] <= 2022)]
summary = ood.groupby('Model')['PRAUC'].agg(['mean','std','min','max']).round(3)
print(summary)

print(f"\nDone. Full results at: {out_csv}")
