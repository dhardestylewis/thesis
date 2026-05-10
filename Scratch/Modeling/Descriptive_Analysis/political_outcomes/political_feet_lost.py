"""
Calculate Political Feet Lost (Net of Compatibility Constraints)
Calculates the true height concession by first capping requested height to the legal
Austin compatibility limit.
"""
import pandas as pd
import numpy as np

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
ADJ_PATH = rf"{OUT_DIR}\adjacent_parcel_universe.csv"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"

print("Loading data...")
adj = pd.read_csv(ADJ_PATH, low_memory=False)
master = pd.read_csv(MASTER_PATH, low_memory=False)

cases = master.dropna(subset=["Final_Zoning", "latitude", "longitude"]).copy()
cases["case_number"] = cases["case_number"].str.strip()
case_coords = cases[["case_number", "latitude", "longitude"]].drop_duplicates("case_number")

adj["case_number"] = adj["case_number"].str.strip()
adj = adj.merge(case_coords, on="case_number", how="inner")
adj["is_single_family"] = (adj["lui_general_land_use"] == 100).astype(int)
sf_parcels = adj[adj["is_single_family"] == 1].copy()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c * 3280.84

sf_parcels["dist_ft"] = haversine(
    sf_parcels["latitude"], sf_parcels["longitude"],
    sf_parcels["p_lat"], sf_parcels["p_lon"]
)

nearest_sf = sf_parcels.groupby("case_number").agg(nearest_sf_ft=("dist_ft", "min")).reset_index()

def calc_compat_height(dist):
    if pd.isna(dist): return 999
    if dist <= 50: return 30
    if dist <= 100: return 40
    if dist <= 300: return 40 + ((dist - 100) / 10)
    if dist <= 540: return 60 + ((dist - 300) / 4)
    return 999

nearest_sf["compat_limit_ft"] = nearest_sf["nearest_sf_ft"].apply(calc_compat_height)

BASE_HEIGHTS = {
    "SF-1": 35, "SF-2": 35, "SF-3": 35, "SF-4A": 35, "SF-4B": 35, "SF-5": 35, "SF-6": 35,
    "TF": 35, "RR": 35, "LA": 35,
    "MF-1": 40, "MF-2": 40, "MF-3": 40, "MF-4": 60, "MF-5": 60, "MF-6": 90,
    "NO": 35, "LO": 40, "GO": 60, "LR": 40, "GR": 60, "CS": 60, "CS-1": 60, "CG": 60, "CR": 60, "CH": 60,
    "LI": 60, "MI": 90, "HI": 90,
    "CBD": 120, "DMU": 120, "TOD": 60, "MU": 60, "PUD": 60
}

OVERLAY_STRIP = __import__("re").compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
def get_base_height(z):
    if not isinstance(z, str): return np.nan
    base = OVERLAY_STRIP.sub("", z.strip().upper()).strip("-")
    return BASE_HEIGHTS.get(base, np.nan)

cases["req_base_height"] = cases["Requested_Zoning"].apply(get_base_height)
cases["fin_base_height"] = cases["Final_Zoning"].apply(get_base_height)

cases = cases.merge(nearest_sf, on="case_number", how="left")
cases["compat_limit_ft"] = cases["compat_limit_ft"].fillna(999)

# 1. Effective Requested Height (Bounded by Compatibility)
cases["effective_req_height"] = cases.apply(
    lambda row: min(row["req_base_height"], row["compat_limit_ft"]) if pd.notna(row["req_base_height"]) else np.nan,
    axis=1
)

# 2. Effective Final Height (Bounded by Compatibility)
cases["effective_fin_height"] = cases.apply(
    lambda row: min(row["fin_base_height"], row["compat_limit_ft"]) if pd.notna(row["fin_base_height"]) else np.nan,
    axis=1
)

# 3. Political Height Lost (Concession *after* Compatibility is priced in)
cases["political_feet_lost"] = cases["effective_req_height"] - cases["effective_fin_height"]

PET_PATH = rf"{OUT_DIR}\petition_intensity_corrected.csv"
pet = pd.read_csv(PET_PATH)
pet["case_number"] = pet["case_number"].str.strip()
cases = cases.merge(pet[["case_number", "label_valid_protest"]], on="case_number", how="left")
cases["label_valid_protest"] = cases["label_valid_protest"].fillna(0)

# Cases that took ANY base downgrade
cases["z_changed"] = cases["Requested_Zoning"].str.strip() != cases["Final_Zoning"].str.strip()
cases["base_downgraded"] = (cases["political_feet_lost"] > 0) & cases["z_changed"]

print("\n" + "="*60)
print("TRUE POLITICAL FEET LOST (Net of Spatial Compatibility Constraints)")
print("="*60)

res = cases[cases["base_downgraded"]].groupby("label_valid_protest").agg(
    cases_with_political_concession=("case_number", "count"),
    avg_political_feet_lost=("political_feet_lost", "mean")
).round(2)
res.index = ["Normal Downgrades (No Valid Protest)", "Protest-Induced Downgrades (>=20%)"]

print(res.to_string())
