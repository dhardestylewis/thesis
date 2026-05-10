import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
from catboost import CatBoostClassifier, Pool
import shap
import warnings
import sys

warnings.filterwarnings('ignore')

sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML")
from run_causal_ml_sweep import load_fully_hydrated_data

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts"

def main():
    print("1. Loading Fully Hydrated Annualized Data Matrix...")
    df, features, categorical_features = load_fully_hydrated_data()

    thresh = 20
    df = df.dropna(subset=["exact_geometric_petition_pct", "latitude", "longitude"])
    y = (df["exact_geometric_petition_pct"] >= thresh).astype(int)
    X = df[features]
    
    print("2. Training CatBoostClassifier on Annualized Data...")
    model = CatBoostClassifier(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        verbose=0, 
        random_seed=42, 
        auto_class_weights='Balanced',
        task_type="GPU"
    )
    
    pool = Pool(X, y, cat_features=categorical_features)
    model.fit(pool)
    
    print("3. Executing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool) # (N, F)
    
    # Calculate global importance to pick top 4 non-coordinate features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    top_features = []
    for idx in sorted_idx:
        feat_name = features[idx]
        if feat_name not in ["latitude", "longitude"] and len(top_features) < 9:
            top_features.append((idx, feat_name))
    
    print(f"Top structural drivers to map: {[f[1] for f in top_features]}")
    
    print("4. Constructing GeoDataFrame...")
    geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Reproject to Web Mercator (EPSG:3857) for contextily basemaps
    gdf = gdf.to_crs(epsg=3857)
    
    # Add SHAP values to the GeoDataFrame
    for idx, feat_name in top_features:
        gdf[f"shap_{feat_name}"] = shap_values[:, idx]
        
    print("5. Rendering Spatial Maps...")
    fig, axes = plt.subplots(3, 3, figsize=(24, 24), dpi=300)
    axes = axes.flatten()
    
    # Compute GLOBAL symmetric color bounds so all subplots are directly comparable
    global_vmax = 0
    for idx, feat_name in top_features:
        feat_vmax = np.abs(gdf[f"shap_{feat_name}"]).max()
        if feat_vmax > global_vmax:
            global_vmax = feat_vmax
            
    global_vmax = global_vmax * 0.8 # Cap at 80% of absolute max for better color spread
    global_vmin = -global_vmax
    
    for i, (idx, feat_name) in enumerate(top_features):
        ax = axes[i]
        
        # Plot points using the global scale
        scatter = gdf.plot(
            column=f"shap_{feat_name}",
            cmap="RdBu_r", # Red represents positive SHAP (higher protest risk), Blue is negative (lower risk)
            ax=ax,
            markersize=15,
            alpha=0.8,
            vmin=global_vmin,
            vmax=global_vmax,
            legend=True,
            legend_kwds={'label': f"SHAP Log-Odds ({feat_name})", 'orientation': "horizontal", 'pad': 0.02, 'shrink': 0.8}
        )
        
        # Add basemap (CartoDB Positron for clean academic look)
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        
        ax.set_axis_off()
        clean_title = feat_name.replace('_', ' ').title()
        ax.set_title(f"Geography of Attribution:\n{clean_title}", fontsize=14, weight="bold")

    plt.suptitle("Spatial Distribution of Causal Attribution (NIMBY Protest ≥ 20%)", fontsize=20, weight="bold", y=1.02)
    plt.tight_layout()
    
    out_path = rf"{OUT_DIR}\causal_shap_geography_map.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Map saved to {out_path}")

if __name__ == "__main__":
    main()
