import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
import geopandas as gpd

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    model_y_multi = CatBoostRegressor(iterations=200, depth=5, loss_function='MultiRMSE', verbose=0)
    model_t = CatBoostClassifier(iterations=200, depth=5, verbose=0)
    model_y_bin = CatBoostRegressor(iterations=200, depth=5, verbose=0)
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.multioutput import MultiOutputRegressor
    model_y_multi = MultiOutputRegressor(GradientBoostingRegressor(max_depth=4, n_estimators=100))
    model_t = GradientBoostingClassifier(max_depth=4, n_estimators=100)
    model_y_bin = GradientBoostingRegressor(max_depth=4, n_estimators=100)

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
DOSE_THRESHOLD = 0.20

print("Loading historical panel...", flush=True)
panel_path = ROOT / "Data/Panel/biweekly_panel_patched.csv"
if not panel_path.exists():
    panel_path = ROOT / "Data/Panel/biweekly_panel.csv"
df = pd.read_csv(panel_path, low_memory=False)

zoning_df = pd.read_csv(ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv", low_memory=False)
zoning_df['start'] = pd.to_datetime(zoning_df['application_start_date'], errors='coerce')
zoning_df['end'] = pd.to_datetime(zoning_df['status_date'], errors='coerce')
zoning_df['days_to_resolution'] = (zoning_df['end'] - zoning_df['start']).dt.days.clip(0, 3650)
zoning_dates = zoning_df[['case_number', 'days_to_resolution']].drop_duplicates('case_number')

status_df = pd.read_csv(ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_case_statuses.csv", low_memory=False)

print("Collapsing panel...", flush=True)
cs = df.groupby('case_number').agg({
    'cumulative_unofficial_protest_intensity': 'max',
    'Delta_Approved_Height': 'last',
    'Delta_Requested_Height': 'last',
    'latitude': 'first',
    'longitude': 'first',
    'median_household_income': 'first',
    'race_white': 'first',
    'renter_share': 'first',
    'local_unemployment_rate': 'first',
    'knn_petition_rate_1km': 'first',
    'dist_petition_rate_lag1': 'first',
    'cumulative_min_signer_dist': 'max',
    'cumulative_signers_outside_200ft': 'max',
    'cumulative_protester_embed_dim1': 'max',
    'cumulative_protester_embed_dim2': 'max',
    'cumulative_petition_attempted': 'max',
    'cumulative_mobilization_failure': 'max'
}).reset_index()

mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0
cs = pd.merge(cs, zoning_dates, on='case_number', how='left')
cs = pd.merge(cs, status_df[['case_number', 'detailed_status']], on='case_number', how='left')

def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    non_zero = x[x > 0]
    if len(non_zero) > 0 and non_zero.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

cs['petition_dose'] = fraction_01(cs['cumulative_unofficial_protest_intensity'])
cs['Height_Attrition'] = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']
cs['Withdrawal_Binary'] = (cs['detailed_status'] == 'Withdrawn').astype(float)

for c in ['median_household_income', 'race_white', 'renter_share']:
    cs[c] = cs[c].fillna(cs[c].median())
for c in ['cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
          'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
          'cumulative_petition_attempted', 'cumulative_mobilization_failure']:
    cs[c] = cs[c].fillna(0.0)

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'renter_share',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1'
]

cs = cs.dropna(subset=confounders + ['Delta_Approved_Height', 'Height_Attrition', 'petition_dose', 'days_to_resolution', 'latitude', 'longitude'])

print("Fitting Causal Forest on historical cases...", flush=True)
X = cs[confounders].values
D = cs['petition_dose'].values
D_bin = (D >= DOSE_THRESHOLD).astype(float)

surv_mask = ~cs['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
cs_surv = cs[surv_mask]
X_surv = cs_surv[confounders].values
Y_surv_joint = cs_surv[['Height_Attrition', 'days_to_resolution']].values
D_bin_surv = (cs_surv['petition_dose'] >= DOSE_THRESHOLD).astype(float).values
Y_withd = cs['Withdrawal_Binary'].values

cf_joint = CausalForestDML(
    model_y=model_y_multi, model_t=model_t,
    discrete_treatment=True, n_estimators=100,
    cv=StratifiedKFold(n_splits=2), random_state=42
)
cf_withd = CausalForestDML(
    model_y=model_y_bin, model_t=model_t,
    discrete_treatment=True, n_estimators=100,
    cv=StratifiedKFold(n_splits=2), random_state=42
)

cf_joint.fit(Y_surv_joint, D_bin_surv, X=X_surv)
cf_withd.fit(Y_withd, D_bin, X=X)

print("Training KNN Demographic Interpolator...", flush=True)
knn_X = cs[['latitude', 'longitude']].values
knn_targets = ['median_household_income', 'race_white', 'renter_share', 'knn_petition_rate_1km', 'dist_petition_rate_lag1']
knn_Y = cs[knn_targets].values
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(knn_X, knn_Y)

print("Loading 300k+ Land Database Parcels...", flush=True)
ldb_path = ROOT / "Data/CoA_Open_Data/Land_Database_2021.geojson"
gdf = gpd.read_file(ldb_path)
if gdf.crs and gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

centroids = gdf.geometry.centroid
gdf['latitude'] = centroids.y
gdf['longitude'] = centroids.x
gdf = gdf.dropna(subset=['latitude', 'longitude'])

print("Interpolating demographics for all parcels...", flush=True)
demo_preds = knn.predict(gdf[['latitude', 'longitude']].values)
for idx, col in enumerate(knn_targets):
    gdf[col] = demo_preds[:, idx]

def simulate_scenario(gdf, height_delta):
    print(f"Simulating +{height_delta} ft scenario...", flush=True)
    gdf['Delta_Requested_Height'] = height_delta
    X_city = gdf[confounders].values
    
    chunk_size = 50000
    delay_preds = []
    withd_preds = []
    
    for i in range(0, len(X_city), chunk_size):
        X_chunk = X_city[i:i+chunk_size]
        cate_multi = cf_joint.effect(X_chunk)
        cate_w = cf_withd.effect(X_chunk)
        delay_preds.append(cate_multi[:, 1])
        withd_preds.append(cate_w)
        
    delay = np.clip(np.concatenate(delay_preds), 0, 3650)
    withd = np.clip(np.concatenate(withd_preds), 0.0, 1.0)
    return delay, withd

# SCENARIOS
scenarios = [
    (0.0, "Use-Change (+0 ft)"),
    (25.0, "Missing Middle (+25 ft)"),
    (55.0, "Transit Corridor (+55 ft)")
]

results = []
for h, label in scenarios:
    delay, withd = simulate_scenario(gdf, h)
    results.append({
        'Scenario': label,
        'Delay Mean': np.mean(delay),
        'Delay Median': np.median(delay),
        'Hurdle Mean': np.mean(withd) * 100,
        'Hurdle Median': np.median(withd) * 100
    })

res_df = pd.DataFrame(results)

latex_code = f"""\\begin{{table}}[H]
\\centering
\\caption[Citywide Simulated Resistance Penalties]{{\\textbf{{Citywide Simulated Resistance Penalties by Planning Scenario.}} Expected processing delay penalty and hurdle risk (probability of petition-induced withdrawal) if subjected to a 20\\% statutory neighborhood petition, simulated across all $N={len(gdf):,}$ addressable Austin parcels. The scenarios represent common entitlement requests: a strict use-change with no physical expansion (+0 ft), a "missing middle" medium-density upzone (+25 ft), and a mid-rise transit corridor rezoning (+55 ft).}}
\\label{{tab:citywide_simulation}}
\\begin{{tabular}}{{l cc cc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{Delay Penalty (Days)}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Hurdle Risk (\\%)}}}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
\\textbf{{Scenario}} & \\textbf{{Mean}} & \\textbf{{Median}} & \\textbf{{Mean}} & \\textbf{{Median}} \\\\
\\midrule
{res_df.iloc[0]['Scenario']} & {res_df.iloc[0]['Delay Mean']:.0f} & {res_df.iloc[0]['Delay Median']:.0f} & {res_df.iloc[0]['Hurdle Mean']:.1f}\\% & {res_df.iloc[0]['Hurdle Median']:.1f}\\% \\\\
{res_df.iloc[1]['Scenario']} & {res_df.iloc[1]['Delay Mean']:.0f} & {res_df.iloc[1]['Delay Median']:.0f} & {res_df.iloc[1]['Hurdle Mean']:.1f}\\% & {res_df.iloc[1]['Hurdle Median']:.1f}\\% \\\\
{res_df.iloc[2]['Scenario']} & {res_df.iloc[2]['Delay Mean']:.0f} & {res_df.iloc[2]['Delay Median']:.0f} & {res_df.iloc[2]['Hurdle Mean']:.1f}\\% & {res_df.iloc[2]['Hurdle Median']:.1f}\\% \\\\
\\bottomrule
\\multicolumn{{5}}{{l}}{{\\footnotesize Generated via continuous spatial causal inference across all municipal parcels.}}\\\\
\\end{{tabular}}
\\end{{table}}
"""

OUT_TEX = ROOT / "Thesis_Draft/GSAPP_Final_Submission/Tables/chapter5_attribution/tbl_ch5_01b_citywide_stats.tex"
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TEX, "w") as f:
    f.write(latex_code)

print("Generated Citywide Scenarios Table!")
