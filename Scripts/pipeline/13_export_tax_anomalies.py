import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

BASE = r"c:\Users\dhl\data\Thesis\thesis"
PANEL_PATH = os.path.join(BASE, "Data", "Panel", "biweekly_cradle_to_grave_panel.csv")
OUT_PATH = os.path.join(BASE, "Data", "Cleaned", "travis_county_tax_anomalies.csv")

def export_anomalies():
    print(f"Loading full 1.15 GB lifecycle panel from {PANEL_PATH}...")
    
    # Load all cases, filtering for only stabilized periods to find mature mispricings
    # Note: We must load everything in chunks to avoid memory explosion, or just load the columns we need.
    cols = [
        "case_number", "period_seq", "appraised_value", "phase_stabilization",
        "mortgage_rate_30yr", "total_population", "cumulative_petition_pct",
        "cumulative_council_hearings", "permit_review_delay_days", "permit_total_job_valuation",
        "land_acres", "building_age"
    ]
    
    # Load into memory (only ~12 columns is very fast and low memory)
    df = pd.read_csv(PANEL_PATH, usecols=cols)
    
    # Filter for stabilization and drop NAs in target/macro
    df_stab = df[(df["phase_stabilization"] == 1) & (df["appraised_value"] > 1000)].dropna(subset=["appraised_value", "mortgage_rate_30yr"]).copy()
    
    print(f"Isolated {len(df_stab):,} biweekly periods across all stabilized projects.")
    
    features = [
        "mortgage_rate_30yr", "total_population", "cumulative_petition_pct",
        "cumulative_council_hearings", "permit_review_delay_days", "permit_total_job_valuation",
        "building_age", "land_acres"
    ]
    
    X = df_stab[features].fillna(0)
    y = df_stab["appraised_value"]
    
    print("Training Fundamental Value Regressor on ALL stabilized properties...")
    # Use 100 trees for full dataset to ensure stability
    reg = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    reg.fit(X, y)
    
    print("Calculating Fundamental Predictions and Z-Scores...")
    df_stab["predicted_value"] = reg.predict(X)
    
    # Require strictly positive predictions
    df_stab = df_stab[df_stab["predicted_value"] > 1000]
    
    # Log-ratio error
    df_stab["log_error"] = np.log(df_stab["predicted_value"] / df_stab["appraised_value"])
    
    # Fit normal distribution to the log_error
    mu, std = norm.fit(df_stab["log_error"])
    
    # Calculate Z-Scores
    df_stab["z_score"] = (df_stab["log_error"] - mu) / std
    
    print("Aggregating properties by case number to find persistent anomalies...")
    # Group by case number to collapse the temporal biweekly periods into single project-level summaries
    mispricings = df_stab.groupby("case_number").agg({
        "appraised_value": "mean",
        "predicted_value": "mean",
        "z_score": "mean",
        "permit_total_job_valuation": "max",
        "land_acres": "max",
        "building_age": "max",
        "cumulative_petition_pct": "max"
    }).reset_index()
    
    mispricings["absolute_z_score"] = np.abs(mispricings["z_score"])
    mispricings["anomaly_type"] = np.where(mispricings["z_score"] > 3.0, "Undervalued by EARS", 
                                  np.where(mispricings["z_score"] < -3.0, "Overvalued by EARS", "Normal"))
    
    # Isolate true anomalies
    anomalies = mispricings[mispricings["absolute_z_score"] >= 3.0].sort_values("absolute_z_score", ascending=False)
    
    print(f"\nDiscovered {len(anomalies)} structural mispricings (> 3 Sigma).")
    
    # Format and save
    out_cols = [
        "case_number", "anomaly_type", "z_score", "appraised_value", "predicted_value", 
        "permit_total_job_valuation", "land_acres", "building_age", "cumulative_petition_pct"
    ]
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    anomalies[out_cols].to_csv(OUT_PATH, index=False)
    print(f"Exported anomaly registry to: {OUT_PATH}")

    # Display Top 10
    print("\n=======================================================")
    print(" TOP 10 MOST EXTREME TAX ANOMALIES IN AUSTIN")
    print("=======================================================")
    for idx, row in anomalies.head(10).iterrows():
        print(f"Case: {row['case_number']:<15} | Z: {row['z_score']:>6.2f} | Type: {row['anomaly_type']:<20}")
        print(f"  EARS Value: ${row['appraised_value']:,.0f} | Model: ${row['predicted_value']:,.0f}")
        print(f"  Drivers:    Permit=${row['permit_total_job_valuation']:,.0f}, Acres={row['land_acres']:.1f}\n")

if __name__ == "__main__":
    export_anomalies()
