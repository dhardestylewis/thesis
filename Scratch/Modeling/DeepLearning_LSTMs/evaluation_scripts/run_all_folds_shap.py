"""
All-Folds GradientSHAP: retrain each expanding window fold, run attribution,
save per-fold feature importance. Output: temporal attribution heatmap.
Runs on GPU.
"""
import os, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from captum.attr import GradientShap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
OUT_DIR    = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

FEATS = [
    "proposed_max_height_ft","existing_max_height_ft","height_delta",
    "proposed_max_far","existing_max_far",
    "period_seq","bw_sin","bw_cos",
    "council_hearings_this_period","cumulative_council_hearings",
    "commission_hearings_this_period","cumulative_commission_hearings",
    "cumulative_petition_events","cumulative_petition_count","cumulative_petition_pct",
    "Remand_Count","market_value","building_age","land_acres",
    "total_population","median_household_income","renter_share","rent_burden",
    "affordability_proxy","race_white","median_age",
    "mortgage_rate_30yr","mortgage_rate_30yr_momentum",
    "treasury_10yr_yield","treasury_10yr_yield_momentum",
    "fed_funds_rate","fed_funds_rate_momentum",
    "local_unemployment_rate","local_unemployment_rate_momentum",
    "knn_petition_rate_1km","dist_petition_rate_lag1"
]

FEAT_LABELS = {
    "existing_max_height_ft":       "Existing Height",
    "dist_petition_rate_lag1":      "Spatial Protest Lag",
    "proposed_max_height_ft":       "Proposed Height",
    "local_unemployment_rate":      "Unemployment Rate",
    "cumulative_petition_events":   "Cumul. Petition Events",
    "proposed_max_far":             "Proposed FAR",
    "existing_max_far":             "Existing FAR",
    "period_seq":                   "Process Age",
    "cumulative_commission_hearings":"Cumul. Commission Hearings",
    "bw_sin":                       "Seasonality",
    "affordability_proxy":          "Affordability Proxy",
    "cumulative_petition_count":    "Cumul. Petition Count",
    "mortgage_rate_30yr_momentum":  "Mortgage Rate Momentum",
    "commission_hearings_this_period":"Commission Hearings",
    "cumulative_petition_pct":      "Cumul. Petition Pct",
    "height_delta":                 "Height Delta",
    "market_value":                 "Market Value",
    "building_age":                 "Building Age",
    "rent_burden":                  "Rent Burden",
    "race_white":                   "Pct White",
    "median_household_income":      "Median Income",
    "renter_share":                 "Renter Share",
    "land_acres":                   "Land (Acres)",
    "total_population":             "Population",
    "median_age":                   "Median Age",
    "mortgage_rate_30yr":           "30yr Mortgage Rate",
    "treasury_10yr_yield":          "10yr Treasury Yield",
    "treasury_10yr_yield_momentum": "Treasury Yield Momentum",
    "fed_funds_rate":               "Fed Funds Rate",
    "fed_funds_rate_momentum":      "Fed Funds Momentum",
    "local_unemployment_rate_momentum": "Unemp. Momentum",
    "cumulative_council_hearings":  "Cumul. Council Hearings",
    "council_hearings_this_period": "Council Hearings",
    "knn_petition_rate_1km":        "KNN Petition Rate 1km",
    "Remand_Count":                 "Remand Count",
    "bw_cos":                       "Seasonality (cos)",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading panel...")
df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
df_raw = df_raw.sort_values(["case_number","period_seq"])
df_raw["proposed_max_height_ft"] = df_raw["proposed_max_height_ft"].fillna(0)
df_raw["existing_max_height_ft"] = df_raw["existing_max_height_ft"].fillna(0)
df_raw["height_delta"] = df_raw["proposed_max_height_ft"] - df_raw["existing_max_height_ft"]
df_raw["period_start"] = pd.to_datetime(df_raw["period_start"], errors="coerce")
df_raw = df_raw.dropna(subset=["period_start"])
df_raw["eval_year"] = df_raw["period_start"].dt.year
window = 52
df_raw["target"] = df_raw.groupby("case_number")["petition_event"].transform(
    lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].shift(-1)
)
df_raw["target"] = df_raw["target"].fillna(0).astype(int)
df_raw = df_raw[df_raw["period_seq"] > 0]
df_raw[FEATS] = df_raw[FEATS].fillna(0)
scaler = StandardScaler()
df_raw[FEATS] = scaler.fit_transform(df_raw[FEATS])

class LSTMHazardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out)

class DS(Dataset):
    def __init__(self, df):
        self.groups = list(df.groupby("case_number"))
    def __len__(self): return len(self.groups)
    def __getitem__(self, i):
        _, g = self.groups[i]
        g = g.sort_values("period_seq")
        return (torch.tensor(g[FEATS].values, dtype=torch.float32),
                torch.tensor(g["target"].values, dtype=torch.float32).unsqueeze(1))

def collate(batch):
    xs=[b[0] for b in batch]; ys=[b[1] for b in batch]
    lens=torch.tensor([len(x) for x in xs])
    xp=pad_sequence(xs,batch_first=True); yp=pad_sequence(ys,batch_first=True)
    mask=torch.zeros(yp.shape[:2],dtype=torch.bool)
    for i,l in enumerate(lens): mask[i,:l]=True
    return xp, yp, lens, mask

all_fold_attrs = {}
test_years = range(2016, 2025)

for test_year in test_years:
    print(f"\n--- Fold {test_year} ---", flush=True)
    train_df = df_raw[df_raw["eval_year"] < test_year]
    test_df  = df_raw[df_raw["eval_year"] == test_year]
    if len(test_df) == 0 or test_df["target"].sum() == 0:
        print(f"  Skipping {test_year} (no events)", flush=True)
        continue

    weights_path = f"fold_{test_year}_model.pt"
    model = LSTMHazardModel(input_dim=len(FEATS)).to(device)

    if os.path.exists(weights_path):
        print(f"  Loading saved weights...", flush=True)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        train_pos = train_df["target"].sum()
        pos_weight = torch.tensor([(len(train_df)-train_pos)/max(train_pos,1)]).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loader     = DataLoader(DS(train_df), batch_size=512, shuffle=True, collate_fn=collate)
        model.train()
        for epoch in range(10):
            loss_sum=0
            for bx,by,lens,mask in loader:
                bx,by,mask=bx.to(device),by.to(device),mask.to(device)
                optimizer.zero_grad()
                loss=criterion(model(bx,lens)[mask],by[mask])
                loss.backward(); optimizer.step()
                loss_sum+=loss.item()
            print(f"  Epoch {epoch+1}/10 | Loss: {loss_sum/len(loader):.4f}", flush=True)
        torch.save(model.state_dict(), weights_path)

    # Build tensors
    test_groups = list(test_df.groupby("case_number"))
    MAX_LEN = max(len(g) for _,g in test_groups)
    def pad_case(g):
        x = torch.tensor(g.sort_values("period_seq")[FEATS].values, dtype=torch.float32)
        return torch.cat([x, torch.zeros(MAX_LEN-len(x), len(FEATS))], dim=0)
    X = torch.stack([pad_case(g) for _,g in test_groups]).to(device)
    lens = torch.tensor([min(len(g),MAX_LEN) for _,g in test_groups])
    baseline = torch.zeros_like(X)

    model.train()
    def fwd(x): return model(x, lens)[:, -1, 0]
    gs = GradientShap(fwd)
    t0 = time.time()
    attrs = gs.attribute(X, baseline, n_samples=50, stdevs=0.1)
    elapsed = time.time() - t0
    mean_attr = attrs.abs().mean(dim=[0,1]).cpu().detach().numpy()
    all_fold_attrs[test_year] = mean_attr
    print(f"  Attribution done in {elapsed:.1f}s", flush=True)

    # Save per-fold CSV
    pd.DataFrame({"feature": FEATS, "mean_abs_attr": mean_attr}).sort_values(
        "mean_abs_attr", ascending=False
    ).to_csv(os.path.join(OUT_DIR, f"shap_fold_{test_year}.csv"), index=False)

# ── Temporal Heatmap ─────────────────────────────────────────────────────────
print("\nBuilding temporal attribution heatmap...", flush=True)
years = sorted(all_fold_attrs.keys())
attr_matrix = np.array([all_fold_attrs[y] for y in years])  # [years x features]

# Normalize each year row to sum to 1 (relative importance)
attr_norm = attr_matrix / attr_matrix.sum(axis=1, keepdims=True)

labels = [FEAT_LABELS.get(f, f) for f in FEATS]
df_heat = pd.DataFrame(attr_norm, index=years, columns=labels)

# Keep only top 15 features by overall mean
top_feats = df_heat.mean().nlargest(15).index.tolist()
df_heat = df_heat[top_feats]

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(df_heat.T, cmap="YlOrRd", linewidths=0.4, linecolor="white",
            annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax,
            cbar_kws={"label": "Normalized Attribution (% of total)"})
ax.set_xlabel("Test Year (Walk-Forward Fold)", fontsize=12, fontweight="bold")
ax.set_ylabel("")
ax.set_title("GradientSHAP Feature Attribution — Temporal Drift (2016-2024)\nDynamic LSTM Hazard Model | 2-Year Petition Horizon",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "shap_temporal_heatmap.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Heatmap saved: {out_path}")
print("All done.")
