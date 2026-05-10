import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedGroupKFold
import os
import sys

sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML")
from run_causal_ml_sweep import load_fully_hydrated_data

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts"

def main():
    print("1. Loading Fully Hydrated Annualized Data...")
    df, features, categorical_features = load_fully_hydrated_data()
    df["filing_year"] = pd.to_datetime(df["application_start_date"]).dt.year
    df = df.dropna(subset=["exact_geometric_petition_pct", "latitude", "longitude", "filing_year"])
    df["target"] = (df["exact_geometric_petition_pct"] >= 20).astype(int)
    
    model_df = df[["case_number", "target", "filing_year"] + features].copy().dropna()
    
    print("2. Generating Out-Of-Time Predictions via Expanding Window...")
    
    out_of_fold_preds = np.full(len(model_df), np.nan)
    years_to_evaluate = sorted([y for y in model_df["filing_year"].unique() if y >= 2018])
    
    model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=0, random_seed=42, auto_class_weights='Balanced')
    
    for yr in years_to_evaluate:
        train_mask = model_df["filing_year"] < yr
        test_mask = model_df["filing_year"] == yr
        
        train = model_df[train_mask]
        test = model_df[test_mask]
        
        if train["target"].sum() < 2 or test["target"].sum() == 0:
            continue
            
        model.fit(Pool(train[features], train["target"], cat_features=categorical_features))
        preds = model.predict_proba(test[features])[:, 1]
        model_df.loc[test_mask, "oof_pred"] = preds
        
    model_df = model_df.dropna(subset=["oof_pred"])
    
    print("3. Classifying Errors...")
    # Calculate Brier score for size mapping
    model_df["squared_error"] = (model_df["target"] - model_df["oof_pred"])**2
    
    # Classify as False Positive, False Negative, True Positive, True Negative
    threshold = 0.5
    model_df["pred_class"] = (model_df["oof_pred"] >= threshold).astype(int)
    
    def classify(row):
        if row["target"] == 1 and row["pred_class"] == 1: return "True Positive (Caught Protest)"
        if row["target"] == 1 and row["pred_class"] == 0: return "False Negative (Missed Protest)"
        if row["target"] == 0 and row["pred_class"] == 1: return "False Positive (False Alarm)"
        return "True Negative (Correct Pass)"
        
    model_df["classification"] = model_df.apply(classify, axis=1)
    
    print("4. Constructing GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        model_df, 
        geometry=gpd.points_from_xy(model_df.longitude, model_df.latitude),
        crs="EPSG:4326"
    )
    # Project to Web Mercator for Contextily
    gdf = gdf.to_crs(epsg=3857)
    
    print("5. Rendering Spatial Error Map...")
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    
    # Plot True Negatives (small grey dots, background)
    tn = gdf[gdf["classification"] == "True Negative (Correct Pass)"]
    tn.plot(ax=ax, color='lightgray', alpha=0.3, markersize=10, label="Correct Non-Protest (TN)")
    
    # Plot False Positives (orange, medium)
    fp = gdf[gdf["classification"] == "False Positive (False Alarm)"]
    fp.plot(ax=ax, color='orange', alpha=0.6, markersize=30, label="False Alarm (FP)")
    
    # Plot True Positives (blue, large)
    tp = gdf[gdf["classification"] == "True Positive (Caught Protest)"]
    tp.plot(ax=ax, color='royalblue', alpha=0.8, markersize=60, edgecolor='black', label="Caught Protest (TP)")
    
    # Plot False Negatives (red, very large)
    fn = gdf[gdf["classification"] == "False Negative (Missed Protest)"]
    fn.plot(ax=ax, color='crimson', alpha=0.9, markersize=80, marker='X', edgecolor='black', label="Missed Protest (FN)")
    
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.legend(loc="upper left", title="Model Calibration (Threshold=0.5)", fontsize=10, title_fontsize=12, framealpha=0.9)
    plt.title("Geography of Structural Predictability:\nWhere Does the Causal Baseline Fail?", fontsize=16, weight="bold")
    plt.tight_layout()
    
    out_img = os.path.join(OUT_DIR, "spatial_error_map_individual.png")
    plt.savefig(out_img, bbox_inches="tight")
    print(f"Spatial error map saved to {out_img}")

if __name__ == "__main__":
    main()
