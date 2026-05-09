import pandas as pd
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
SPATIAL_PATH = rf"{OUT_DIR}\spatial_attribution_2024.csv"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"

def main():
    print(f"1. Loading Base Biweekly Panel from {PANEL_PATH}...")
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["case_number"] = panel["case_number"].astype(str).str.strip()
    
    # 1. Merge Geometric Petition Pct
    print("2. Merging Exact Geometric Petition Intensity...")
    geom_df = pd.read_csv(GEOM_PATH)
    geom_df["case_number"] = geom_df["case_number"].astype(str).str.strip()
    # If the column already exists, drop it to avoid _x, _y duplication
    if "label_exact_geometric_petition_pct" in panel.columns:
        panel = panel.drop(columns=["label_exact_geometric_petition_pct"])
    panel = panel.merge(geom_df[["case_number", "label_exact_geometric_petition_pct"]], on="case_number", how="left")
    panel["label_exact_geometric_petition_pct"] = panel["label_exact_geometric_petition_pct"].fillna(0)
    
    # 2. Merge Spatial Blight
    print("3. Merging Spatial Blight Indices...")
    spatial_df = pd.read_csv(SPATIAL_PATH, low_memory=False)
    spatial_df["case_number"] = spatial_df["case_number"].astype(str).str.strip()
    spatial_df = spatial_df.drop_duplicates(subset=["case_number"])
    spatial_cols = ["archetype_pct_Architectural", "archetype_pct_Bureaucratic", "archetype_pct_Economic", "archetype_pct_Spatial_Gravity"]
    
    for c in spatial_cols:
        if c in panel.columns:
            panel = panel.drop(columns=[c])
            
    panel = panel.merge(spatial_df[["case_number"] + spatial_cols], on="case_number", how="left")
    for c in spatial_cols:
        panel[c] = panel[c].fillna(0)
        
    # 3. Explicitly Drop Leakage (NLP / Remands)
    print("4. Removing Target Leakage...")
    nlp_cols = ["Aggregate_Sentiment", "Opposition_Volume", "Support_Volume", "Remand_Count"]
    for c in nlp_cols:
        if c in panel.columns:
            panel = panel.drop(columns=[c])
            
    print(f"5. Saving Leakage-Free Hydrated Panel to {PANEL_PATH}...")
    panel.to_csv(PANEL_PATH, index=False)
    print(f"Done! New panel has {panel.shape[0]} rows and {panel.shape[1]} columns.")

if __name__ == "__main__":
    main()
