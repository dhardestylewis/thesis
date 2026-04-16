"""
identify_regimes.py
===================
Uses a Gaussian Mixture Model (GMM) to discover latent structural
environments (regimes) across 2,000+ zoning cases for robust 
Invariant Risk Minimization / V-REx.

Outputs: advanced_regime_assignments.csv
"""
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os
import re

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_DIR = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(ROOT, "Analysis", "Results")
os.makedirs(OUT_DIR, exist_ok=True)

ZONING_PATH = os.path.join(DATA_DIR, "Zoning_Cases", "Processed_Data", "multi_parcel_closed_2018_2025.csv")
IRM_ENV_PATH = os.path.join(ROOT, "Analysis", "Results", "irm_environment_assignments.csv")
OUT_PATH = os.path.join(ROOT, "Analysis", "Results", "advanced_regime_assignments.csv")

def identify_regimes(n_regimes=5):
    print("Loading multi-parcel cases for Regime Discovery...")
    zoning = pd.read_csv(ZONING_PATH, low_memory=False)
    
    zoning['year'] = pd.to_datetime(zoning['APPLICATION_START_DATE'], errors='coerce').dt.year
    
    # We cluster based on Year and Location (Lat/Lon)
    model_df = zoning.dropna(subset=['year', 'LATITUDE', 'LONGITUDE']).copy()
    
    print(f"Total valid cases for GMM clustering: {len(model_df)}")
    
    features = ['year', 'LATITUDE', 'LONGITUDE']
    X = model_df[features].values
    
    print(f"Fitting Gaussian Mixture Model (K={n_regimes})...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)
    regimes = gmm.fit_predict(X_scaled)
    
    model_df['regime_id'] = [f"Regime_{r}" for r in regimes]
    
    # Merge the identified Regime (Environment) metadata onto the existing assignment file
    # This prevents PyTorch from loading ONLY 1 valid environment
    irm = pd.read_csv(IRM_ENV_PATH)
    irm['CASE_NUMBER'] = irm['CASE_NUMBER'].astype(str).str.strip()
    model_df['CASE_NUMBER'] = model_df['CASE_NUMBER'].astype(str).str.strip()
    
    # Merge using strict strings
    adv = irm.merge(model_df[['CASE_NUMBER', 'regime_id', 'LATITUDE', 'LONGITUDE']], on='CASE_NUMBER', how='left')
    
    # Fill any truly orphaned rows
    adv['regime_id'] = adv['regime_id'].fillna('Regime_Unknown')
    adv['zoning_delta_intensity'] = 1.0  # Optional fallback
    
    adv = adv.rename(columns={'regime_id': 'env_id'})
    
    adv.to_csv(OUT_PATH, index=False)
    print(f"\nSaved Advanced Regime assignments to: {OUT_PATH}")
    print("\nEnvironment distribution:")
    print(adv['env_id'].value_counts())

if __name__ == "__main__":
    identify_regimes()
