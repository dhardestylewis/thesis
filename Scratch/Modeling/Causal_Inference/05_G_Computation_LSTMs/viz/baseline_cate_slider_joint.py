import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel_patched.csv"

def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    non_zero_x = x[x > 0]
    if len(non_zero_x) > 0 and non_zero_x.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

print('Loading panel...', flush=True)
if os.path.exists(PANEL_PATH):
    df = pd.read_csv(PANEL_PATH, low_memory=False)
else:
    df = pd.read_csv(ROOT / "Data/Panel/biweekly_panel.csv", low_memory=False)

print('Loading raw zoning dates...', flush=True)
zoning_path = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv"
zoning_df = pd.read_csv(zoning_path, low_memory=False)
zoning_df['start'] = pd.to_datetime(zoning_df['application_start_date'], errors='coerce')
zoning_df['end'] = pd.to_datetime(zoning_df['status_date'], errors='coerce')
zoning_df['days_to_resolution'] = (zoning_df['end'] - zoning_df['start']).dt.days
zoning_df['days_to_resolution'] = zoning_df['days_to_resolution'].clip(0, 3650)
zoning_dates = zoning_df[['case_number', 'days_to_resolution']].drop_duplicates('case_number')

print('Loading Socrata case statuses...', flush=True)
status_path = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_case_statuses.csv"
status_df = pd.read_csv(status_path, low_memory=False)

print('Collapsing to cross-sectional...', flush=True)
cs = df.groupby('case_number').agg({
    'cumulative_unofficial_protest_intensity': 'max',
    'Delta_Approved_Height': 'last',
    'Delta_Requested_Height': 'last',
    'latitude': 'first',
    'longitude': 'first',
    'median_household_income': 'first',
    'race_white': 'first',
    'renter_share': 'first',
    'year': 'first',
    'cumulative_min_signer_dist': 'max',
    'cumulative_signers_outside_200ft': 'max',
    'cumulative_protester_embed_dim1': 'max',
    'cumulative_protester_embed_dim2': 'max',
    'cumulative_petition_attempted': 'max',
    'cumulative_mobilization_failure': 'max'
}).reset_index()

# SURVIVOR BIAS PATCH
mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0

cs = pd.merge(cs, zoning_dates, on='case_number', how='left')
cs = pd.merge(cs, status_df[['case_number', 'detailed_status']], on='case_number', how='left')

cs['petition_dose'] = fraction_01(cs['cumulative_unofficial_protest_intensity'])
cs['Height_Attrition'] = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']
cs['Withdrawal_Binary'] = (cs['detailed_status'] == 'Withdrawn').astype(float)

# Impute minor missingness
for c in ['median_household_income', 'race_white', 'renter_share']:
    cs[c] = cs[c].fillna(cs[c].median())
for c in ['cumulative_min_signer_dist', 'cumulative_signers_outside_200ft', 'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2', 'cumulative_petition_attempted', 'cumulative_mobilization_failure']:
    cs[c] = cs[c].fillna(0.0)

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude', 
    'median_household_income', 'race_white', 'renter_share',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

cs = cs.dropna(subset=confounders + ['Delta_Approved_Height', 'Height_Attrition', 'petition_dose', 'days_to_resolution', 'year'])

X = cs[confounders].values
D = cs['petition_dose'].values

surv_mask = ~cs['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
cs_surv = cs[surv_mask]
X_surv = cs_surv[confounders].values
Y_surv_joint = cs_surv[['Height_Attrition', 'days_to_resolution']].values
Y_withd = cs['Withdrawal_Binary'].values

thresholds = [0.001] + list(np.arange(0.05, 0.40, 0.05))
surfaces = []

print('Computing Joint CATE surfaces across thresholds...', flush=True)

# Define models - using LGBM because we want this script to be fast and runnable locally
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    model_y_multi = CatBoostRegressor(iterations=100, depth=4, loss_function='MultiRMSE', verbose=0)
    model_t = CatBoostClassifier(iterations=100, depth=4, verbose=0)
    model_y_bin = CatBoostRegressor(iterations=100, depth=4, verbose=0)
except ImportError:
    from sklearn.multioutput import MultiOutputRegressor
    model_y_multi = MultiOutputRegressor(LGBMRegressor(max_depth=3, min_child_samples=5, n_estimators=100))
    model_t = LGBMClassifier(max_depth=3, min_child_samples=5, n_estimators=100)
    model_y_bin = LGBMRegressor(max_depth=3, min_child_samples=5, n_estimators=100)

from sklearn.model_selection import StratifiedKFold

for t in thresholds:
    print(f'  Testing Threshold >= {t:.3f}...')
    D_bin = (D >= t).astype(float)
    D_bin_surv = (cs_surv['petition_dose'] >= t).astype(float).values
    
    if D_bin_surv.sum() < 2 or D_bin.sum() < 2:
        print('   Not enough treated cases. Empty surface.')
        continue
        
    cf_joint = CausalForestDML(
        model_y=model_y_multi,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=100,
        cv=StratifiedKFold(n_splits=2),
        random_state=42
    )
    
    cf_withd = CausalForestDML(
        model_y=model_y_bin,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=100,
        cv=StratifiedKFold(n_splits=2),
        random_state=42
    )
    
    try:
        cf_joint.fit(Y_surv_joint, D_bin_surv, X=X_surv)
        cate_multi = cf_joint.effect(X) 
        cate_height = cate_multi[:, 0]
        cate_delay = cate_multi[:, 1]
        
        cf_withd.fit(Y_withd, D_bin, X=X)
        cate_withd = cf_withd.effect(X)
        
        # Clip wild DML outliers caused by low propensity score support
        cate_height = np.clip(cate_height, -500, 1500)
        cate_delay = np.clip(cate_delay, -365, 3650)
        cate_withd = np.clip(cate_withd, -1.0, 1.0)
        
        surf_df = cs[['latitude', 'longitude']].copy()
        surf_df['cate_height'] = cate_height
        surf_df['cate_delay'] = cate_delay
        surf_df['cate_withd'] = cate_withd
        
        surfaces.append({
            'threshold': t,
            'data': surf_df
        })
        print(f'   Mean Height CATE: {cate_height.mean():.4f}, Delay CATE: {cate_delay.mean():.4f}, Withd CATE: {cate_withd.mean():.4f}')
    except Exception as e:
        print(f'   Error fitting forest: {e}')

fig = go.Figure()

if surfaces:
    # Normalize globally so colors/opacities are consistent across frames
    all_delays = np.concatenate([s['data']['cate_delay'].dropna().values for s in surfaces])
    all_withds = np.concatenate([s['data']['cate_withd'].dropna().values for s in surfaces])
    d_min, d_max = np.percentile(all_delays, 5), np.percentile(all_delays, 95)
    w_min, w_max = np.percentile(all_withds, 5), np.percentile(all_withds, 95)
    norm_delay = mcolors.Normalize(vmin=d_min, vmax=d_max)
    
    # Delay is predominantly positive (bureaucratic friction/heat). 
    # A divergent map like RdBu_r is misleading here because the center (white) 
    # won't be zero, causing positive delays to appear "blue" (which implies a speedup).
    # 'inferno' (dark purple to bright yellow) perfectly encodes friction on a dark theme.
    cmap = cm.get_cmap('inferno')
    
    for i, surf in enumerate(surfaces):
        df_surf = surf['data']
        thresh = surf['threshold']
        
        delays = df_surf['cate_delay'].values
        withds = df_surf['cate_withd'].values
        
        # Graveyard threshold: >10% increased probability of withdrawal
        is_dead = withds > 0.10
        surv_mask = ~is_dead
        
        # 1. Surviving Trace (Friction Layer)
        delays_surv = delays[surv_mask]
        colors_surv = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.9)' for r, g, b, _ in cmap(norm_delay(delays_surv))]
        
        fig.add_trace(go.Scatter3d(
            x=df_surf.loc[surv_mask, 'longitude'],
            y=df_surf.loc[surv_mask, 'latitude'],
            z=df_surf.loc[surv_mask, 'cate_height'],
            mode='markers',
            marker=dict(size=4, color=colors_surv, line=dict(width=0)),
            name=f'Threshold >= {thresh:.3f} (Survives)',
            visible=(i==0),
            customdata=delays_surv,
            hovertemplate='Lon: %{x:.4f}<br>Lat: %{y:.4f}<br>Height CATE: %{z:.2f} ft<br>Delay CATE: %{customdata:.1f} days<extra>Survived</extra>'
        ))
        
        # 2. Graveyard Trace (Lethal Layer)
        fig.add_trace(go.Scatter3d(
            x=df_surf.loc[is_dead, 'longitude'],
            y=df_surf.loc[is_dead, 'latitude'],
            z=df_surf.loc[is_dead, 'cate_height'],
            mode='markers',
            marker=dict(size=4, color='rgba(128, 128, 128, 0.5)', line=dict(width=0)),
            name=f'Threshold >= {thresh:.3f} (Killed)',
            visible=(i==0),
            hovertemplate='Lon: %{x:.4f}<br>Lat: %{y:.4f}<br>Height CATE: %{z:.2f} ft<br>Outcome: <b style="color:red">Killed (Withdrawn)</b><extra></extra>'
        ))

    # Fake trace for the Delay CATE colorbar
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            colorscale='Inferno',
            cmin=d_min,
            cmax=d_max,
            colorbar=dict(title='Delay CATE (days)')
        ),
        hoverinfo='none',
        showlegend=False,
        visible=True
    ))

    steps = []
    num_traces = (len(surfaces) * 2) + 1
    
    for i, surf in enumerate(surfaces):
        thresh = surf['threshold']
        step_visible = [False] * num_traces
        
        # Turn on the two traces for this threshold
        step_visible[2 * i] = True       # Surviving Trace
        step_visible[(2 * i) + 1] = True # Graveyard Trace
        step_visible[-1] = True          # Fake Colorbar Trace
        
        step = dict(
            method='update',
            args=[{'visible': step_visible},
                  {'title': f'Joint Causal Targets (Height, Delay, Withdrawal) for Dose >= {thresh:.3f}'}],
            label=f'{thresh:.3f}' if thresh > 0.001 else '> 0.0'
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': 'Petition Dose Threshold: '},
        pad={'t': 50},
        steps=steps
    )]

    first_t = surfaces[0]['threshold']
    
    # Setup consistent bounds
    lon_min, lon_max = cs['longitude'].min() - 0.01, cs['longitude'].max() + 0.01
    lat_min, lat_max = cs['latitude'].min() - 0.01, cs['latitude'].max() + 0.01
    z_min, z_max = cs['Height_Attrition'].min() - 5, cs['Height_Attrition'].max() + 5

    fig.update_layout(
        sliders=sliders,
        title=f'Joint Causal Targets (Height, Delay, Withdrawal) for Dose >= {first_t:.3f}',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Height CATE (Extrusion)',
            xaxis=dict(range=[lon_min, lon_max]),
            yaxis=dict(range=[lat_min, lat_max]),
            zaxis=dict(range=[z_min, z_max]),
            aspectmode='cube'
        ),
        template='plotly_dark'
    )

    out_path = r'C:\Users\dhl\.gemini\antigravity\brain\1632e32a-ef31-4422-854b-ea7296224fe1\cate_spatial_dose_slider_joint.html'
    fig.write_html(out_path)
    print(f'\nSaved Joint Slider HTML to {out_path}')
else:
    print("No surfaces were generated!")
