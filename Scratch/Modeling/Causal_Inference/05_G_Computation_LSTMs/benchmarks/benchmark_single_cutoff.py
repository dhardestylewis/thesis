import time
import torch
import pandas as pd
import numpy as np
import re
from run_causal_lstm_era_slider import precompute_tensors, get_train_tensors_from_cache, train_era_model

print("Loading dataset...")
df = pd.read_csv(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv", low_memory=False)
master = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", low_memory=False)

OVERLAY_STRIP = re.compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)

master["case_number"] = master["case_number"].astype(str).str.strip()
master["req_int"] = master["Requested_Zoning"].apply(get_int)
master["fin_int"] = master["Final_Zoning"].apply(get_int)
master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)
df["case_number"] = df["case_number"].astype(str).str.strip()
df = df.merge(master[["case_number", "t_downgrade"]].drop_duplicates("case_number"), on="case_number", how="left")
df["t_downgrade"] = df["t_downgrade"].fillna(0)

df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
features = ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period", "cumulative_petition_pct", "bw_sin", "bw_cos"]
for f in features: df[f] = df[f].fillna(0)
df["resolved"] = df["resolved"].fillna(0).astype(int)
df["council_hearings_this_period"] = df["council_hearings_this_period"].fillna(0).astype(float)
df["commission_hearings_this_period"] = df["commission_hearings_this_period"].fillna(0).astype(float)

norm_dict = {}
for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
    mean_v = df[f].mean()
    std_v = df[f].std()
    df[f] = (df[f] - mean_v) / (std_v + 1e-8)

print("Precomputing tensor cache...")
cache = precompute_tensors(df, features, 30)

era_dt = pd.to_datetime("2020-01-01")
print(f"Extracting tensors for cutoff: {era_dt.strftime('%Y-%m-%d')}...")
dataset, K = get_train_tensors_from_cache(cache, era_dt)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dataset Size: {len(dataset)} sequences.")
print(f"Executing Training on {device.upper()}...")

t0 = time.time()
model = train_era_model(dataset, features, device)
t1 = time.time()

print(f"\n[BENCHMARK] Training a single cutoff model took exactly: {t1-t0:.2f} seconds.")
