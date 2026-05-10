import os
import pandas as pd
import numpy as np

BASE_DIR = r"c:\Users\dhl\data\Thesis\thesis"
ANNUAL_CSV = os.path.join(BASE_DIR, "Scratch", "Modeling", "Causal_Inference", "04_T_Learner_ML", "model_ready_zoning_supplemented.csv")
BIWEEKLY_CSV = os.path.join(BASE_DIR, "Scratch", "Modeling", "Causal_Inference", "05_G_Computation_LSTMs", "biweekly_panel.csv")
OUT_CSV = os.path.join(BASE_DIR, "Scratch", "Modeling", "Causal_Inference", "04_T_Learner_ML", "model_ready_zoning_supplemented.csv")

def build_supplemented_panel():
    print(f"Loading Annual Panel: {ANNUAL_CSV}")
    annual = pd.read_csv(ANNUAL_CSV, low_memory=False)
    
    print(f"Loading Biweekly Panel: {BIWEEKLY_CSV}")
    # Load only necessary columns to save memory
    biweekly_cols = [
        # Panel ID / sequencing
        "case_number", "period_seq",
        # Demographics (ACS)
        "total_population", "median_household_income",
        "renter_share", "rent_burden", "affordability_proxy",
        "race_white", "race_black", "race_hispanic", "median_age",
        # Macro drivers
        "mortgage_rate_30yr", "local_unemployment_rate",
        "fed_funds_rate", "treasury_10yr_yield",
        "mortgage_rate_30yr_momentum", "local_unemployment_rate_momentum",
        "fed_funds_rate_momentum", "treasury_10yr_yield_momentum",
        # Spatial contagion / gravity
        "knn_petition_rate_1km", "dist_petition_rate_lag1",
        "active_cases_1km", "active_cases_2km",
        # EARS parcel economics (TCAD)
        "market_value", "land_market_value", "improvement_market_value",
        "land_acres", "building_age", "improvement_sq_ft", "improvement_ratio",
        # Zoning intensity / height features (PDF-extracted)
        "pdf_requested_height_ft", "pdf_proposed_height_ft",
        "pdf_requested_max_far", "pdf_compatibility_height_ft",
        # NLP filing-day signals (leakage-safe at period_seq==1)
        "nlp_document_count", "nlp_oppose_hits", "nlp_traffic_hits", "nlp_density_hits",
        # Developer/process velocity
        "hearing_velocity_3p", "max_opponent_experience",
    ]
    biweekly = pd.read_csv(BIWEEKLY_CSV, usecols=biweekly_cols)
    
    # We only want the features AT THE TIME OF FILING (period_seq == 1)
    baseline_biweekly = biweekly[biweekly["period_seq"] == 1].copy()
    baseline_biweekly = baseline_biweekly.drop(columns=["period_seq"])
    
    print(f"Merging {len(baseline_biweekly)} baseline rows onto the {len(annual)} annual cases...")
    
    # Drop any columns that were previously merged from biweekly to avoid conflicts
    stale_cols = [c for c in annual.columns if c in baseline_biweekly.columns and c != "case_number"]
    if stale_cols:
        print(f"Dropping {len(stale_cols)} stale biweekly columns before re-merge...")
        annual = annual.drop(columns=stale_cols)
    
    # Merge onto the annual panel
    supplemented = annual.merge(baseline_biweekly, on="case_number", how="left")
    
    # Drop rows where we couldn't match (should be negligible)
    missing = supplemented["mortgage_rate_30yr"].isna().sum()
    print(f"Cases missing biweekly baseline data: {missing} / {len(supplemented)}")
    
    supplemented.to_csv(OUT_CSV, index=False)
    print(f"Saved Supplemented Annual Panel to: {OUT_CSV}")
    print(f"Final shape: {supplemented.shape}")

if __name__ == "__main__":
    build_supplemented_panel()
