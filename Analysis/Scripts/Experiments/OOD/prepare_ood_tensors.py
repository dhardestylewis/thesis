"""
prepare_ood_tensors.py
======================
Builds PyTorch tensors for ICP invariance testing.

Methodology:
  - Environments: Austin Council Districts 1-10 (genuine political subdivisions)
  - Cross-validation: Expanding-window temporal CV (5 folds)
  - Sampling: Full population with valid council_district (no subsampling)
  - Target: Binary protest indicator (binarized for classification)
  - Features: 23 numeric + PCA-compressed categoricals

Author: Daniel Hardesty Lewis
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import os, json, time

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_v3.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "Analysis", "Data", "Tensors")
os.makedirs(OUT_DIR, exist_ok=True)

# Expanding-window CV folds (train up to year T, test on year T+1)
CV_FOLDS = [
    {'train_end': 2019, 'test_year': 2020},
    {'train_end': 2020, 'test_year': 2021},
    {'train_end': 2021, 'test_year': 2022},
    {'train_end': 2022, 'test_year': 2023},
    {'train_end': 2023, 'test_year': 2024},
]

N_PCA_COMPONENTS = 100
MAX_SAMPLE_PER_FOLD = 500000  # per-fold cap to prevent memory issues

# Features — exclude spatial leakage variables
NUMERIC_FEATURES = [
    'total_market_value', 'deed_acreage', 'improvement_sq_ft',
    'improvement_market_value', 'land_market_value',
    'appraised_value', 'assessed_value', 'taxable_value',
    'prior_year_taxable_value', 'new_construction_value',
    'total_exemption_amount',
    'improvement_ratio', 'value_density', 'land_to_total_ratio', 'imprv_to_land_ratio',
    'tax_gap', 'yoy_value_change', 'building_age',
    'protest_nearby_area_pct',
    'year',
]

CATEGORICAL_FEATURES = [
    'property_category_code', 'lui_general_land_use',
    'lui_land_use', 'subcategory_code', 'zoning_code',
    'homesite_flag', 'exemption_flag_hs', 'special_use_flag', 'freeze_flag',
]
# NOTE: council_district is the ENVIRONMENT, not a feature


def main():
    t0 = time.time()
    
    # ── 1. Load full panel ─────────────────────────────────────────────────
    print("Loading full panel...")
    cols = [
        'total_market_value', 'deed_acreage', 'improvement_sq_ft',
        'improvement_market_value', 'land_market_value',
        'appraised_value', 'assessed_value', 'taxable_value',
        'prior_year_taxable_value', 'new_construction_value',
        'property_category_code', 'council_district', 'lui_general_land_use',
        'lui_land_use', 'subcategory_code', 'zoning_code',
        'year', 'year_built', 'homesite_flag', 'exemption_flag_hs',
        'protest', 'protest_nearby_area_pct',
        'total_exemption_amount', 'special_use_flag', 'freeze_flag',
        'latitude', 'longitude',
    ]
    
    chunk_iter = pd.read_csv(PANEL_PATH, usecols=cols, chunksize=500000, low_memory=False)
    panel_chunks = []
    for i, chunk in enumerate(chunk_iter):
        chunk = chunk[chunk['year'] <= 2024].copy()
        chunk['improvement_sq_ft'] = pd.to_numeric(chunk['improvement_sq_ft'], errors='coerce')
        panel_chunks.append(chunk)
        if (i + 1) % 4 == 0:
            print(f"  Loaded {(i+1)*500000:,} rows...")
    
    panel = pd.concat(panel_chunks, ignore_index=True)
    del panel_chunks
    print(f"  Total panel rows: {len(panel):,}")
    
    # ── 2. Environment assignment: Council Districts (fast spatial join) ──
    import geopandas as gpd
    from matplotlib.path import Path as MplPath
    
    DISTRICTS_PATH = os.path.join(PROJECT_DIR, "Data", "GIS", "council_districts.geojson")
    print("\nFast spatial join: assigning council districts...")
    
    districts = gpd.read_file(DISTRICTS_PATH)
    print(f"  District GeoJSON columns: {list(districts.columns)}")
    
    # Find the district number column dynamically (prefer 'COUNCIL_DI' over 'OBJECTID')
    dist_col = None
    for c in districts.columns:
        cu = c.upper()
        if 'COUNCIL' in cu and ('DIST' in cu or 'DI' in cu):
            dist_col = c
            break
    if dist_col is None:
        for c in districts.columns:
            if 'DISTRICT' in c.upper():
                dist_col = c
                break
    if dist_col is None:
        dist_col = districts.columns[1]
    print(f"  Using district column: {dist_col}")
    print(f"  Sample values: {districts[dist_col].tolist()}")
    
    # Get valid lat/lon rows
    valid_mask = panel['latitude'].notna() & panel['longitude'].notna()
    lons = panel.loc[valid_mask, 'longitude'].values
    lats = panel.loc[valid_mask, 'latitude'].values
    coords = np.column_stack([lons, lats])
    print(f"  Parcels with coordinates: {len(coords):,} of {len(panel):,}")
    
    # Fast point-in-polygon: iterate 10 polygons, batch-check all points
    district_assignments = np.full(len(coords), -1, dtype=int)
    for _, row in districts.iterrows():
        d_num = int(row[dist_col])
        geom = row.geometry
        # Handle MultiPolygon
        if geom.geom_type == 'MultiPolygon':
            polys = list(geom.geoms)
        else:
            polys = [geom]
        
        for poly in polys:
            ext_coords = np.array(poly.exterior.coords)
            path = MplPath(ext_coords)
            inside = path.contains_points(coords)
            district_assignments[inside] = d_num
    
    panel.loc[valid_mask, 'council_district_geo'] = district_assignments
    panel['council_district_geo'] = panel['council_district_geo'].fillna(-1).astype(int)
    
    # Filter to valid districts 1-10
    panel = panel[panel['council_district_geo'].between(1, 10)].copy()
    panel['env_id'] = 'District_' + panel['council_district_geo'].astype(str)
    panel['env_label'] = panel['council_district_geo'] - 1  # 0-indexed for PyTorch
    
    print(f"  Parcels assigned to districts: {len(panel):,}")
    print(f"  Environment distribution:")
    print(panel['env_id'].value_counts().sort_index().to_string())
    
    # ── 3. Feature engineering ─────────────────────────────────────────────
    print("\nEngineering features...")
    panel['improvement_ratio'] = panel['improvement_sq_ft'] / (panel['deed_acreage'] * 43560 + 1e-6)
    panel['value_density'] = panel['total_market_value'] / (panel['deed_acreage'] * 43560 + 1e-6)
    panel['land_to_total_ratio'] = panel['land_market_value'] / (panel['total_market_value'] + 1e-6)
    panel['imprv_to_land_ratio'] = panel['improvement_market_value'] / (panel['land_market_value'] + 1e-6)
    panel['tax_gap'] = (panel['appraised_value'] - panel['assessed_value']).fillna(0)
    panel['yoy_value_change'] = (panel['taxable_value'] - panel['prior_year_taxable_value']).fillna(0)
    panel['building_age'] = panel['year'] - panel['year_built'].fillna(panel['year'])
    
    for c in ['protest_nearby_area_pct', 'new_construction_value', 'total_exemption_amount',
              'improvement_market_value', 'land_market_value', 'appraised_value',
              'assessed_value', 'taxable_value', 'prior_year_taxable_value']:
        panel[c] = pd.to_numeric(panel[c], errors='coerce').fillna(0)
    
    for col in NUMERIC_FEATURES:
        panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in CATEGORICAL_FEATURES:
        panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)
    
    panel['protest'] = panel['protest'].fillna(0).astype(int)
    
    print(f"  Protest=1: {(panel['protest']==1).sum():,} ({100*(panel['protest']==1).mean():.2f}%)")
    print(f"  Year range: {panel['year'].min()} - {panel['year'].max()}")
    
    # ── 4. Expanding-window CV: build tensors per fold ─────────────────────
    print(f"\n{'='*80}")
    print(f"EXPANDING-WINDOW CV: {len(CV_FOLDS)} folds")
    print(f"{'='*80}")
    
    all_fold_metadata = []
    
    for fold_i, fold in enumerate(CV_FOLDS):
        fold_name = f"fold{fold_i+1}"
        fold_dir = os.path.join(OUT_DIR, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        
        train_mask = panel['year'] <= fold['train_end']
        test_mask = panel['year'] == fold['test_year']
        
        train_df = panel[train_mask].copy()
        test_df = panel[test_mask].copy()
        
        # Stratified subsample if too large
        if len(train_df) > MAX_SAMPLE_PER_FOLD:
            # Keep all positives, sample negatives
            pos = train_df[train_df['protest'] == 1]
            neg = train_df[train_df['protest'] == 0].sample(
                n=min(MAX_SAMPLE_PER_FOLD - len(pos), len(train_df[train_df['protest'] == 0])),
                random_state=42
            )
            train_df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        if len(test_df) > MAX_SAMPLE_PER_FOLD:
            test_df = test_df.sample(n=MAX_SAMPLE_PER_FOLD, random_state=42).reset_index(drop=True)
        
        print(f"\n  Fold {fold_i+1}: Train <=  {fold['train_end']} ({len(train_df):,} rows), Test = {fold['test_year']} ({len(test_df):,} rows)")
        print(f"    Train protest rate: {train_df['protest'].mean():.4f}")
        print(f"    Train envs: {train_df['env_id'].nunique()}")
        
        # Fit scaler + encoder on train, transform both (sparse to avoid OOM)
        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        
        X_train_num = scaler.fit_transform(train_df[NUMERIC_FEATURES])
        X_train_cat_raw = encoder.fit_transform(train_df[CATEGORICAL_FEATURES])
        
        X_test_num = scaler.transform(test_df[NUMERIC_FEATURES])
        X_test_cat_raw = encoder.transform(test_df[CATEGORICAL_FEATURES])
        
        # PCA compression of OHE categoricals
        n_comp = min(N_PCA_COMPONENTS, X_train_cat_raw.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_train_cat = svd.fit_transform(X_train_cat_raw)
        X_test_cat = svd.transform(X_test_cat_raw)
        explained = svd.explained_variance_ratio_.sum()
        print(f"    PCA: {X_train_cat_raw.shape[1]} OHE -> {n_comp} components ({explained:.1%} variance)")
        
        X_train = np.hstack([X_train_num, X_train_cat]).astype(np.float32)
        X_test = np.hstack([X_test_num, X_test_cat]).astype(np.float32)
        y_train = train_df['protest'].values.astype(np.float32)
        y_test = test_df['protest'].values.astype(np.float32)
        envs_train = train_df['env_label'].values.astype(np.int64)
        env_ids_train = train_df['env_id'].values
        
        # Save tensors
        torch.save(torch.from_numpy(X_train), os.path.join(fold_dir, "X_train.pt"))
        torch.save(torch.from_numpy(y_train), os.path.join(fold_dir, "y_train.pt"))
        torch.save(torch.from_numpy(envs_train), os.path.join(fold_dir, "envs_train.pt"))
        torch.save(torch.from_numpy(X_test), os.path.join(fold_dir, "X_test.pt"))
        torch.save(torch.from_numpy(y_test), os.path.join(fold_dir, "y_test.pt"))
        np.save(os.path.join(fold_dir, "env_ids_train.npy"), np.array(env_ids_train, dtype=str))
        
        cat_pca_names = [f'cat_pca_{i}' for i in range(n_comp)]
        full_feature_names = list(NUMERIC_FEATURES) + cat_pca_names
        unique_envs = sorted([e for e in train_df['env_id'].unique()])
        
        fold_meta = {
            'fold_name': fold_name,
            'train_end_year': fold['train_end'],
            'test_year': fold['test_year'],
            'train_size': len(train_df),
            'test_size': len(test_df),
            'n_features': X_train.shape[1],
            'numeric_features': list(NUMERIC_FEATURES),
            'full_feature_names': full_feature_names,
            'unique_envs': unique_envs,
            'n_numeric': len(NUMERIC_FEATURES),
            'n_pca_components': n_comp,
            'pca_explained_variance': float(explained),
            'protest_rate_train': float(train_df['protest'].mean()),
        }
        
        with open(os.path.join(fold_dir, 'metadata.json'), 'w') as f:
            json.dump(fold_meta, f, indent=2)
        
        all_fold_metadata.append(fold_meta)
        print(f"    Saved: {fold_dir} ({X_train.shape[1]} features)")
    
    # Save global metadata
    with open(os.path.join(OUT_DIR, 'cv_metadata.json'), 'w') as f:
        json.dump(all_fold_metadata, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Done in {elapsed:.0f}s. Saved {len(CV_FOLDS)} fold(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
