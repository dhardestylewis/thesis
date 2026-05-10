import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import re

print("Loading Zoning Cases...")
BASE = r"C:\Users\dhl\data\Thesis\thesis\Data"
zoning = pd.read_csv(os.path.join(BASE, "final", "model_ready_zoning_data.csv"), low_memory=False)
zoning["App_Date"] = pd.to_datetime(zoning["App_Date"], errors="coerce").dt.tz_localize(None)
zoning = zoning.dropna(subset=["latitude", "longitude", "App_Date"])
gdf_z = gpd.GeoDataFrame(zoning, geometry=gpd.points_from_xy(zoning.longitude, zoning.latitude), crs="EPSG:4326")

# 1. Process Plan Review Cases (Pre-App)
print("Loading Plan Review Cases...")
pr = pd.read_csv(os.path.join(BASE, "CoA_Open_Data", "Plan_Review_Cases.csv"), low_memory=False)
# Extract lat/lon from Location: POINT (-97.756 30.346)
pr["Location"] = pr["Location"].astype(str)
pr["longitude"] = pr["Location"].str.extract(r'POINT \(([-\d\.]+)')[0].astype(float)
pr["latitude"] = pr["Location"].str.extract(r'POINT \([-\d\.]+\s([-\d\.]+)\)')[0].astype(float)
pr["Issued_Date"] = pd.to_datetime(pr["Issued_Date"], errors="coerce").dt.tz_localize(None)
pr = pr.dropna(subset=["latitude", "longitude", "Issued_Date"])
gdf_pr = gpd.GeoDataFrame(pr, geometry=gpd.points_from_xy(pr.longitude, pr.latitude), crs="EPSG:4326")

# Find if a Zoning Case had a preceding Pre-App within 100m and 2 years before
# We will use BallTree for fast spatial query
tree_pr = BallTree(np.radians(gdf_pr[["latitude", "longitude"]].values), metric="haversine")
had_preapp = []
for idx, row in gdf_z.iterrows():
    # Find all PR cases within 100m (approx 0.001 radians)
    radius_rad = 100 / 6371000
    ind = tree_pr.query_radius(np.radians([[row.latitude, row.longitude]]), r=radius_rad)[0]
    if len(ind) > 0:
        match_dates = gdf_pr.iloc[ind]["Issued_Date"]
        # Check temporal constraint: PR must be before Zoning App, but not more than 3 years before
        valid = match_dates[(match_dates < row.App_Date) & (match_dates > row.App_Date - pd.Timedelta(days=1095))]
        had_preapp.append(1 if len(valid) > 0 else 0)
    else:
        had_preapp.append(0)

gdf_z["had_preapp"] = had_preapp
print(f"Zoning cases with Pre-App matched: {sum(had_preapp)}")

# 2. Process Site Plan Cases
print("Loading Site Plan Cases...")
sp = pd.read_csv(os.path.join(BASE, "CoA_Open_Data", "Site_Plan_Cases.csv"), low_memory=False)
sp["LATITUDE"] = pd.to_numeric(sp["LATITUDE"], errors="coerce")
sp["LONGITUDE"] = pd.to_numeric(sp["LONGITUDE"], errors="coerce")
sp["APPLICATION_START_DATE"] = pd.to_datetime(sp["APPLICATION_START_DATE"], errors="coerce").dt.tz_localize(None)
sp = sp.dropna(subset=["LATITUDE", "LONGITUDE", "APPLICATION_START_DATE"])
gdf_sp = gpd.GeoDataFrame(sp, geometry=gpd.points_from_xy(sp.LONGITUDE, sp.LATITUDE), crs="EPSG:4326")

tree_sp = BallTree(np.radians(gdf_sp[["LATITUDE", "LONGITUDE"]].values), metric="haversine")
days_to_sp = []
for idx, row in gdf_z.iterrows():
    radius_rad = 100 / 6371000
    ind = tree_sp.query_radius(np.radians([[row.latitude, row.longitude]]), r=radius_rad)[0]
    if len(ind) > 0:
        match_dates = gdf_sp.iloc[ind]["APPLICATION_START_DATE"]
        # Site plan must be submitted after Zoning App
        valid = match_dates[match_dates >= row.App_Date]
        if len(valid) > 0:
            delay = (valid.min() - row.App_Date).days
            days_to_sp.append(delay)
        else:
            days_to_sp.append(np.nan)
    else:
        days_to_sp.append(np.nan)

gdf_z["days_to_site_plan"] = days_to_sp

# 3. Process Building Permits
print("Loading Building Permits...")
bp = pd.read_csv(os.path.join(BASE, "CoA_Open_Data", "Issued_Building_Permits.csv"), low_memory=False)
bp["LATITUDE"] = pd.to_numeric(bp["LATITUDE"], errors="coerce")
bp["LONGITUDE"] = pd.to_numeric(bp["LONGITUDE"], errors="coerce")
bp["ISSUE_DATE"] = pd.to_datetime(bp["ISSUE_DATE"], errors="coerce").dt.tz_localize(None)
bp = bp.dropna(subset=["LATITUDE", "LONGITUDE", "ISSUE_DATE"])
# We only care about new construction or large additions
bp = bp[bp["WORK_TYPE"].str.contains("New|Add", na=False, case=False)]
gdf_bp = gpd.GeoDataFrame(bp, geometry=gpd.points_from_xy(bp.LONGITUDE, bp.LATITUDE), crs="EPSG:4326")

tree_bp = BallTree(np.radians(gdf_bp[["LATITUDE", "LONGITUDE"]].values), metric="haversine")
days_to_bp = []
for idx, row in gdf_z.iterrows():
    radius_rad = 100 / 6371000
    ind = tree_bp.query_radius(np.radians([[row.latitude, row.longitude]]), r=radius_rad)[0]
    if len(ind) > 0:
        match_dates = gdf_bp.iloc[ind]["ISSUE_DATE"]
        valid = match_dates[match_dates >= row.App_Date]
        if len(valid) > 0:
            delay = (valid.min() - row.App_Date).days
            days_to_bp.append(delay)
        else:
            days_to_bp.append(np.nan)
    else:
        days_to_bp.append(np.nan)

gdf_z["days_to_building_permit"] = days_to_bp

out_cols = ["case_number", "had_preapp", "days_to_site_plan", "days_to_building_permit"]
out_df = gdf_z[out_cols]
out_path = os.path.join(BASE, "interim", "amanda_spatial_features.csv")
out_df.to_csv(out_path, index=False)
print(f"Saved {len(out_df)} AMANDA spatial features to {out_path}")
