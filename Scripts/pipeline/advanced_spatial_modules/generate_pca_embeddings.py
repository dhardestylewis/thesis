import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import os

def build_pca_embeddings(petitions, tcad, cases_gdf, props=None, out_dir=r"Data/Protest_Petitions"):
    signed_cases = petitions['case_number'].unique()
    print(f"2. Computing localized matrices for {len(signed_cases)} protested cases...")
    
    cases_to_process = cases_gdf[cases_gdf.index.isin(signed_cases)].copy()
    cases_to_process['geometry'] = cases_to_process.geometry.buffer(200)
    
    joined = gpd.sjoin(tcad, cases_to_process.reset_index(), how='inner', predicate='intersects')
    
    case_embeddings = []
    
    for case in cases_to_process.index:
        # All parcels within 200ft
        neighbors = joined[joined['case_number'] == case].index.astype(str).unique()
        if len(neighbors) == 0:
            continue
            
        # Fix float string bug (10003.0 -> '10003') while supporting dashes
        raw_signers = petitions[petitions['case_number'] == case]['tcad_id'].dropna().astype(str)
        signers = set(raw_signers.str.replace(r'\.0$', '', regex=True).str.replace('-', '', regex=False).unique())
        
        protesting_ids = [n for n in neighbors if n in signers]
        
        if len(protesting_ids) == 0:
            case_embeddings.append({'case_number': case, 'protester_embed_dim1': 0, 'protester_embed_dim2': 0, 'protester_embed_dim3': 0, 'protester_embed_dim4': 0})
            continue
            
        # Look up properties
        protesting_props = props.reindex(protesting_ids)
        
        # Build Matrix
        sqft = protesting_props['lui_shape_area'].fillna(0).values
        # One-hot encode basic land uses
        is_sf = (protesting_props['lui_general_land_use'] == 'Single Family').astype(float).values
        is_com = (protesting_props['lui_general_land_use'] == 'Commercial').astype(float).values
        is_mf = (protesting_props['lui_general_land_use'] == 'Multifamily').astype(float).values
        
        # Build the localized matrix for this neighborhood
        local_matrix = np.column_stack([sqft, is_sf, is_com, is_mf])
        
        # Generate the case representation vector (mean pool)
        # To get a single vector representing the protest, we mean pool the local matrix
        case_vector = np.nanmean(local_matrix, axis=0) # 4-dimensional vector
        
        case_embeddings.append({
            'case_number': case,
            'protester_embed_dim1': case_vector[0],
            'protester_embed_dim2': case_vector[1],
            'protester_embed_dim3': case_vector[2],
            'protester_embed_dim4': case_vector[3]
        })
        
    embed_df = pd.DataFrame(case_embeddings).fillna(0)
    
    # We apply PCA ACROSS all cases to ensure the representation space is properly distributed
    print("3. Applying global PCA over the localized case representations...")
    features = embed_df[['protester_embed_dim1', 'protester_embed_dim2', 'protester_embed_dim3', 'protester_embed_dim4']].values
    
    # Fill any NaNs from local division by zero
    features = np.nan_to_num(features, nan=0.0)
    
    # PCA to strictly decorrelate the dimensions
    pca = PCA(n_components=4)
    dense_embeds = pca.fit_transform(features)
    
    embed_df['protester_embed_dim1'] = dense_embeds[:, 0]
    embed_df['protester_embed_dim2'] = dense_embeds[:, 1]
    embed_df['protester_embed_dim3'] = dense_embeds[:, 2]
    embed_df['protester_embed_dim4'] = dense_embeds[:, 3]
    
    print(f"PCA Variance Explained: {pca.explained_variance_ratio_}")
    
    out_path = r'Data/Protest_Petitions/pca_embeddings.csv'
    embed_df.to_csv(out_path, index=False)
    print(f"Saved literal PCA embeddings to {out_path}")
    
    print("4. Merging into Advanced Petition Panel...")
    adv_path = r'Data/Protest_Petitions/advanced_geometric_petition_intensity.csv'
    adv = pd.read_csv(adv_path)
    
    cols_to_drop = [c for c in adv.columns if 'embed_dim' in c]
    adv = adv.drop(columns=cols_to_drop)
    
    merged = pd.merge(adv, embed_df, on='case_number', how='left').fillna(0)
    merged.to_csv(adv_path, index=False)
    print("Merged and saved advanced geometric petition intensity.")

if __name__ == "__main__":
    build_pca_embeddings()
