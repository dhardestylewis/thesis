import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import json

BASE = r"C:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs"
PANEL_PATH = os.path.join(OUT_DIR, "biweekly_panel.csv")
BIPARTITE_PATH = os.path.join(OUT_DIR, "bipartite_edges.csv")

print("1. Loading Panel...")
panel = pd.read_csv(PANEL_PATH, low_memory=False)
panel["period_start"] = pd.to_datetime(panel["period_start"])
# Need T0 and T_end to calculate active cases
zoning_path = os.path.join(BASE, "final", "model_ready_zoning_data.csv")
zoning = pd.read_csv(zoning_path, low_memory=False)
zoning = zoning.rename(columns={"application_start_date": "App_Date"})
zoning["App_Date"] = pd.to_datetime(zoning["App_Date"], errors="coerce").dt.tz_localize(None)

# Estimate T_end from T_vote or censored windows like build_biweekly_panel.py did
zoning["T_vote"] = pd.to_datetime(zoning.get("Final_Council_Date", pd.Series(dtype="datetime64[ns]")).fillna(zoning.get("final_date")).fillna(zoning.get("approval_date")), errors="coerce").dt.tz_localize(None)
zoning["T_end"] = zoning["T_vote"].fillna(zoning["App_Date"] + pd.Timedelta(days=730))

print("2. Calculating Ratios...")
panel["hearing_frequency"] = panel["cumulative_council_hearings"] / panel["period_seq"]
panel["petition_intensity_per_ft"] = panel["cumulative_petition_pct"] / panel["pdf_requested_height_ft"].replace(0, np.nan)
panel["staff_concession_ratio"] = panel.get("pdf_staff_recommends_ht", pd.Series(dtype=float)) / panel["pdf_requested_height_ft"].replace(0, np.nan)

print("3. Calculating Velocities & Lags...")
panel = panel.sort_values(["case_number", "period_seq"])
panel["hearing_velocity_3p"] = panel.groupby("case_number")["cumulative_council_hearings"].diff(3).fillna(0)
panel["petition_velocity_3p"] = panel.groupby("case_number")["cumulative_petition_pct"].diff(3).fillna(0)

print("   -> Engineering causal cumulative lag (shift 1)...")
panel["cumulative_petition_pct_lag1"] = (
    panel.groupby("case_number")["petition_pct_this_period"]
         .apply(lambda s: s.shift(1).fillna(0).cumsum())
         .reset_index(level=0, drop=True)
)

print("4. Calculating Bipartite Opponent Centrality (max_opponent_experience)...")
if os.path.exists(BIPARTITE_PATH):
    edges = pd.read_csv(BIPARTITE_PATH)
    # Get T0 for each case to enforce leakage safety
    case_dates = zoning[["case_number", "App_Date"]].dropna()
    edges = edges.merge(case_dates, left_on="target_case_number", right_on="case_number", how="inner")
    edges = edges.sort_values("App_Date")
    
    # For each owner, at the time of a protest, how many PRIOR protests did they do?
    # We can do this by enumerating their appearance
    edges["owner_prior_protests"] = edges.groupby("owner_name").cumcount()
    
    # Now for each case, what is the MAX prior experience of any of its opponents?
    max_exp = edges.groupby("target_case_number")["owner_prior_protests"].max().reset_index()
    max_exp.rename(columns={"target_case_number": "case_number", "owner_prior_protests": "max_opponent_experience"}, inplace=True)
    
    if "max_opponent_experience" in panel.columns:
        panel.drop(columns=["max_opponent_experience"], inplace=True)
    panel = panel.merge(max_exp, on="case_number", how="left")
    panel["max_opponent_experience"] = panel["max_opponent_experience"].fillna(0).astype(int)
    print(f"   Added max_opponent_experience for {panel['max_opponent_experience'].gt(0).sum()} rows.")
else:
    print("   Bipartite edges not found.")
    panel["max_opponent_experience"] = 0

print("5. Calculating Temporal-Spatial Contagion (Rings & Gravity)...")
# Drop the old 500m feature if it exists to avoid duplication
if "active_cases_500m_t" in panel.columns:
    panel.drop(columns=["active_cases_500m_t"], inplace=True)

# Pre-compute neighbors up to 2km (2000m)
z_valid = zoning.dropna(subset=["latitude", "longitude", "App_Date", "T_end"]).copy()
tree = BallTree(np.radians(z_valid[["latitude", "longitude"]].values), metric="haversine")
max_radius_rad = 2000 / 6371000

# Dictionary mapping case_number to a list of (T0, T_end, distance_meters) for all its neighbors within 2km
neighbors_dict = {}
cases = z_valid["case_number"].values
t0s = z_valid["App_Date"].values
tends = z_valid["T_end"].values

inds, dists = tree.query_radius(np.radians(z_valid[["latitude", "longitude"]].values), r=max_radius_rad, return_distance=True)
for i, case in enumerate(cases):
    # Neighbors excluding self
    neigh_data = []
    for j, dist_rad in zip(inds[i], dists[i]):
        if j != i:
            dist_m = dist_rad * 6371000
            neigh_data.append((t0s[j], tends[j], dist_m))
    neighbors_dict[case] = neigh_data

def get_spatial_features(row):
    case = row["case_number"]
    t = row["period_start"]
    feats = {"100m": 0, "250m": 0, "500m": 0, "1km": 0, "2km": 0, "gravity": 0.0}
    if case not in neighbors_dict:
        return feats
    
    for (t0, tend, dist_m) in neighbors_dict[case]:
        if t0 <= t <= tend:
            if dist_m <= 100: feats["100m"] += 1
            if dist_m <= 250: feats["250m"] += 1
            if dist_m <= 500: feats["500m"] += 1
            if dist_m <= 1000: feats["1km"] += 1
            if dist_m <= 2000: feats["2km"] += 1
            feats["gravity"] += 1.0 / (dist_m + 1.0)
            
    return feats

# Apply function
spatial_results = []
for idx, row in panel.iterrows():
    spatial_results.append(get_spatial_features(row))

spatial_df = pd.DataFrame(spatial_results)
panel["active_cases_100m"] = spatial_df["100m"]
panel["active_cases_250m"] = spatial_df["250m"]
panel["active_cases_500m"] = spatial_df["500m"]
panel["active_cases_1km"] = spatial_df["1km"]
panel["active_cases_2km"] = spatial_df["2km"]
panel["active_gravity_index_t"] = spatial_df["gravity"]

print("6. Saving updated panel...")
panel.to_csv(PANEL_PATH, index=False)
print(f"Saved {len(panel)} rows to {PANEL_PATH}")
