import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from run_causal_lstm_era_slider import precompute_tensors, get_train_tensors_from_cache

# Load data and setup cache
print("Loading data for PRAUC calculation...")
df = pd.read_csv(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv", low_memory=False)
master = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", low_memory=False)
import re
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

for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
    mean_v = df[f].mean()
    std_v = df[f].std()
    df[f] = (df[f] - mean_v) / (std_v + 1e-8)

cache = precompute_tensors(df, features, 30)

# Load the PyTorch LSTM Ensemble
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\causal_lstm_biweekly_ensemble.pt", map_location=device, weights_only=False)
models = checkpoint["models"]

era_str = '2020-12-27'
era_dt = pd.to_datetime(era_str)
dataset, _ = get_train_tensors_from_cache(cache, era_dt)

from run_causal_lstm_era_slider import MultiTaskLSTM
model = MultiTaskLSTM(len(features)).to(device)
model.load_state_dict(models[era_str])
model.eval()

y_surv_true, y_surv_probs = [], []
y_dg_true, y_dg_probs = [], []
y_comm_true, y_comm_preds = [], []
y_counc_true, y_counc_preds = [], []

with torch.no_grad():
    for x, t_surv, d, c1, c2 in dataset:
        x = x.unsqueeze(0).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred_surv, pred_dg, pred_comm, pred_counc = model(x)
        
        # Calculate length based on non-zero padding (or the length of the tensor)
        # Since these aren't batched and padded here, length is just the tensor size
        length = x.shape[1]
        
        # Survival (Classification)
        s_logits = pred_surv.cpu().numpy()[0, :length]
        s_probs = 1 / (1 + np.exp(-s_logits))
        true_s = t_surv.numpy()[:length]
        y_surv_true.extend(true_s)
        y_surv_probs.extend(s_probs)
        
        # Downgrade (Classification)
        dg_logits = pred_dg.cpu().numpy()[0, :length]
        dg_probs = 1 / (1 + np.exp(-dg_logits))
        y_dg_true.append(float(d.item()))
        y_dg_probs.append(float(dg_probs[-1])) # The final step prediction
        
        # Commission (Regression)
        comm_preds = np.maximum(0, pred_comm.cpu().numpy()[0, :length])
        true_comm = c1.numpy()[:length]
        y_comm_true.extend(true_comm)
        y_comm_preds.extend(comm_preds)
        
        # Council (Regression)
        counc_preds = np.maximum(0, pred_counc.cpu().numpy()[0, :length])
        true_counc = c2.numpy()[:length]
        y_counc_true.extend(true_counc)
        y_counc_preds.extend(counc_preds)

y_surv_true = np.array(y_surv_true)
y_surv_probs = np.array(y_surv_probs)
y_dg_true = np.array(y_dg_true)
y_dg_probs = np.array(y_dg_probs)

prauc_surv = average_precision_score(y_surv_true, y_surv_probs)
prauc_dg = average_precision_score(y_dg_true, y_dg_probs)

from sklearn.metrics import mean_squared_error
mse_comm = mean_squared_error(y_comm_true, y_comm_preds)
mse_counc = mean_squared_error(y_counc_true, y_counc_preds)

print("="*50)
print(f"MTL Evaluated on Era {era_str}")
print(f"[Survival] PRAUC: {prauc_surv:.4f} | Baseline: {np.mean(y_surv_true):.4f}")
print(f"[Downgrade] PRAUC: {prauc_dg:.4f} | Baseline: {np.mean(y_dg_true):.4f}")
print(f"[Commission] MSE: {mse_comm:.4f}")
print(f"[Council] MSE: {mse_counc:.4f}")
print("="*50)
