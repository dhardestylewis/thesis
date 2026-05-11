import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor
import plotly.graph_objects as go
import os

PANEL_PATH = r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv'

print('Loading panel...')
df = pd.read_csv(PANEL_PATH, low_memory=False)

def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    non_zero_x = x[x > 0]
    if len(non_zero_x) > 0 and non_zero_x.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

# Collapse panel to cross-sectional (Baseline model only)
cs = df.groupby('case_number').agg({
    'cumulative_unofficial_protest_intensity': 'max',
    'Delta_Approved_Height': 'last',
    'Delta_Requested_Height': 'last', # The missing confounder!
    'latitude': 'first',
    'longitude': 'first',
    'median_household_income': 'first',
    'race_white': 'first',
    'renter_share': 'first',
    'cumulative_min_signer_dist': 'max',
    'cumulative_signers_outside_200ft': 'max',
    'cumulative_protester_embed_dim1': 'max',
    'cumulative_protester_embed_dim2': 'max',
    'cumulative_petition_attempted': 'max',
    'cumulative_mobilization_failure': 'max'
}).reset_index()

# SURVIVOR BIAS PATCH: If Delta_Requested_Height is known but Delta_Approved_Height is missing,
# it means the case was withdrawn or denied (developer got 0 extra height above base zoning).
mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0


cs['petition_dose'] = fraction_01(cs['cumulative_unofficial_protest_intensity'])

# Critical: Added Delta_Requested_Height to the confounder matrix
confounders = ['Delta_Requested_Height', 'latitude', 'longitude', 'median_household_income', 'race_white', 'renter_share']

# Impute minor missingness in census demographics
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

# Drop ANY remaining NaNs to prevent EconML from crashing
cs = cs.dropna(subset=confounders + ['Delta_Approved_Height', 'petition_dose'])

# Calculate Height Attrition (How much of the initial request was lost)
cs['Height_Attrition'] = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']

X = cs[confounders].values

# CONTINUOUS OUTCOME: Height Attrition (Feet negotiated away)
Y = cs['Height_Attrition'].values

thresholds = [0.001] + list(np.arange(0.05, 1.05, 0.05))
surfaces = []

print('Computing Continuous CATE surfaces across thresholds...')
for t in thresholds:
    print(f'  Testing Threshold >= {t:.3f}...')
    D_bin = (cs['petition_dose'] >= t).astype(float).values
    
    if D_bin.sum() < 5:
        print('   Not enough treated cases. Empty surface.')
        surf_df = cs[['latitude', 'longitude']].copy()
        surf_df['cate'] = np.nan
        surfaces.append({
            'threshold': t,
            'data': surf_df
        })
        continue
        
    cf = CausalForestDML(
        model_y=LGBMRegressor(max_depth=3, min_child_samples=5),
        model_t=LGBMClassifier(max_depth=3, min_child_samples=5),
        discrete_treatment=True,
        n_estimators=100,
        random_state=42
    )
    
    try:
        cf.fit(Y, D_bin, X=X)
        cate = cf.effect(X)
        
        surf_df = cs[['latitude', 'longitude']].copy()
        surf_df['cate'] = cate
        
        surfaces.append({
            'threshold': t,
            'data': surf_df
        })
        print(f'   Mean CATE (feet): {cate.mean():.4f}')
    except Exception as e:
        print(f'   Error fitting forest: {e}')

# Create Plotly figure with slider
fig = go.Figure()

# Find global bounds to completely lock the physical geometry grid
valid_cates = [s['data']['cate'].dropna().values for s in surfaces if len(s['data']['cate'].dropna()) > 0]
if valid_cates:
    all_cates = np.concatenate(valid_cates)
    
    # Let's cap outliers at the 2nd and 98th percentile for visual clarity
    z_min = np.percentile(all_cates, 2)
    z_max = np.percentile(all_cates, 98)
    
    # Make the color scale symmetric around 0 so white is exactly 0
    abs_max = max(abs(z_min), abs(z_max))
    z_min, z_max = -abs_max, abs_max
else:
    z_min, z_max = -50, 50

lon_min, lon_max = cs['longitude'].min(), cs['longitude'].max()
lat_min, lat_max = cs['latitude'].min(), cs['latitude'].max()

# Add a tiny buffer so points aren't exactly on the bounding box
z_min -= 1
z_max += 1
lon_min -= 0.01
lon_max += 0.01
lat_min -= 0.01
lat_max += 0.01

for i, surf in enumerate(surfaces):
    df_surf = surf['data']
    thresh = surf['threshold']
    
    fig.add_trace(go.Scatter3d(
        x=df_surf['longitude'],
        y=df_surf['latitude'],
        z=df_surf['cate'],
        mode='markers',
        marker=dict(
            size=4,
            color=df_surf['cate'],
            colorscale='RdBu_r',
            cmin=z_min,
            cmax=z_max,
            colorbar=dict(title='CATE (Height in Feet)')
        ),
        name=f'Threshold >= {thresh:.3f}',
        visible=(i==0)
    ))

steps = []
for i, surf in enumerate(surfaces):
    thresh = surf['threshold']
    step = dict(
        method='update',
        args=[{'visible': [True] + [False] * len(surfaces)},
              {'title': f'Spatial CATE Surface for Petition Dose >= {thresh:.3f} (Height Attrition)'}],
        label=f'{thresh:.3f}' if thresh > 0.001 else '> 0.0'
    )
    step['args'][0]['visible'][i+1] = True
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={'prefix': 'Petition Dose Threshold: '},
    pad={'t': 50},
    steps=steps
)]

if surfaces:
    first_t = surfaces[0]['threshold']
    fig.update_layout(
        sliders=sliders,
        title=f'Spatial CATE Surface for Petition Dose >= {first_t:.3f} (Height Attrition)',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Treatment Effect (Attrition Feet)',
            xaxis=dict(range=[lon_min, lon_max]),
            yaxis=dict(range=[lat_min, lat_max]),
            zaxis=dict(range=[z_min, z_max]),
            aspectmode='cube'
        ),
        template='plotly_dark'
    )

    out_path = r'C:\Users\dhl\.gemini\antigravity\brain\1632e32a-ef31-4422-854b-ea7296224fe1\cate_spatial_dose_slider_attrition_unofficial.html'
    fig.write_html(out_path)
    print(f'\nSaved Continuous Slider HTML to {out_path}')
