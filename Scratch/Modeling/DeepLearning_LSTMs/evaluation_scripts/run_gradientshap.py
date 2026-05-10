"""
Full GradientSHAP attribution on 2024 test fold.
Outputs: feature importance bar chart + raw attribution CSV.
"""
import os, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from captum.attr import GradientShap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PANEL_PATH  = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
WEIGHTS_PATH = "fold_2024_model.pt"
OUT_DIR     = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
TRAIN_YEAR  = 2024

FEATS = [
    "proposed_max_height_ft", "existing_max_height_ft", "height_delta",
    "proposed_max_far", "existing_max_far",
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
    "Remand_Count",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income",
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum",
    "treasury_10yr_yield", "treasury_10yr_yield_momentum",
    "fed_funds_rate", "fed_funds_rate_momentum",
    "local_unemployment_rate", "local_unemployment_rate_momentum",
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load panel ──────────────────────────────────────────────────────────────
print("Loading panel...")
df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
df_raw = df_raw.sort_values(["case_number", "period_seq"])
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

train_df = df_raw[df_raw["eval_year"] < TRAIN_YEAR]
test_df  = df_raw[df_raw["eval_year"] == TRAIN_YEAR]

# ── Model ───────────────────────────────────────────────────────────────────
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

model = LSTMHazardModel(input_dim=len(FEATS)).to(device)

# ── Load or retrain ──────────────────────────────────────────────────────────
if os.path.exists(WEIGHTS_PATH):
    print(f"Loading saved weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
else:
    print(f"No saved weights found — retraining on data < {TRAIN_YEAR}...")
    class DS(Dataset):
        def __init__(self, df):
            self.groups = list(df.groupby("case_number"))
        def __len__(self): return len(self.groups)
        def __getitem__(self, i):
            _, g = self.groups[i]
            g = g.sort_values("period_seq")
            x = torch.tensor(g[FEATS].values, dtype=torch.float32)
            y = torch.tensor(g["target"].values, dtype=torch.float32).unsqueeze(1)
            return x, y

    def collate(batch):
        xs = [b[0] for b in batch]; ys = [b[1] for b in batch]
        lens = torch.tensor([len(x) for x in xs])
        xp = pad_sequence(xs, batch_first=True)
        yp = pad_sequence(ys, batch_first=True)
        mask = torch.zeros(yp.shape[:2], dtype=torch.bool)
        for i, l in enumerate(lens): mask[i, :l] = True
        return xp, yp, lens, mask

    train_pos = train_df["target"].sum()
    pos_weight = torch.tensor([(len(train_df) - train_pos) / max(train_pos, 1)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader     = DataLoader(DS(train_df), batch_size=512, shuffle=True, collate_fn=collate)
    model.train()
    for epoch in range(10):
        loss_sum = 0
        for bx, by, lens, mask in loader:
            bx, by, mask = bx.to(device), by.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx, lens)[mask], by[mask])
            loss.backward(); optimizer.step()
            loss_sum += loss.item()
        print(f"  Epoch {epoch+1}/10 | Loss: {loss_sum/len(loader):.4f}", flush=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)

# ── Build fixed-length tensor for all test cases ─────────────────────────────
print(f"\nBuilding attribution tensors for {len(test_df['case_number'].unique())} test cases...")
test_groups = list(test_df.groupby("case_number"))
MAX_LEN = max(len(g) for _, g in test_groups)

def pad_case(g):
    x = torch.tensor(g.sort_values("period_seq")[FEATS].values, dtype=torch.float32)
    pad = torch.zeros(MAX_LEN - len(x), len(FEATS))
    return torch.cat([x, pad], dim=0)

X_all   = torch.stack([pad_case(g) for _, g in test_groups]).to(device)
lens_all = torch.tensor([min(len(g), MAX_LEN) for _, g in test_groups])
baseline = torch.zeros_like(X_all)

# ── GradientSHAP ─────────────────────────────────────────────────────────────
model.train()  # cuDNN RNN requires train mode for backward

def forward_fn(x):
    return model(x, lens_all)[:, -1, 0]

gs = GradientShap(forward_fn)
print("Running GradientSHAP (n_samples=50)...", flush=True)
t0 = time.time()
attributions = gs.attribute(X_all, baseline, n_samples=50, stdevs=0.1)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

# ── Aggregate: mean |attribution| per feature ────────────────────────────────
mean_attr = attributions.abs().mean(dim=[0, 1]).cpu().detach().numpy()
feat_df = pd.DataFrame({"feature": FEATS, "mean_abs_attr": mean_attr})
feat_df = feat_df.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)

# Save raw
feat_df.to_csv(os.path.join(OUT_DIR, "gradientshap_feature_importance.csv"), index=False)

print("\nTop 15 Features by Mean |GradientSHAP Attribution|:")
print(feat_df.head(15).to_string(index=False))

# ── Plot ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="flare")
fig, ax = plt.subplots(figsize=(10, 8))

FEATURE_LABELS = {
    "knn_petition_rate_1km":        "KNN Petition Rate (1km)",
    "dist_petition_rate_lag1":      "Spatial Protest Density Lag",
    "cumulative_petition_events":   "Cumulative Petition Events",
    "cumulative_petition_count":    "Cumulative Petition Count",
    "cumulative_petition_pct":      "Cumulative Petition Area %",
    "period_seq":                   "Period Sequence (Process Age)",
    "cumulative_council_hearings":  "Cumulative Council Hearings",
    "cumulative_commission_hearings": "Cumulative Commission Hearings",
    "council_hearings_this_period": "Council Hearings (This Period)",
    "commission_hearings_this_period": "Commission Hearings (This Period)",
    "Remand_Count":                 "Remand Count",
    "height_delta":                 "Height Delta (Proposed - Existing)",
    "proposed_max_height_ft":       "Proposed Max Height (ft)",
    "existing_max_height_ft":       "Existing Max Height (ft)",
    "proposed_max_far":             "Proposed Max FAR",
    "existing_max_far":             "Existing Max FAR",
    "market_value":                 "Market Value",
    "building_age":                 "Building Age",
    "land_acres":                   "Land (Acres)",
    "total_population":             "Total Population",
    "median_household_income":      "Median Household Income",
    "renter_share":                 "Renter Share",
    "rent_burden":                  "Rent Burden",
    "affordability_proxy":          "Affordability Proxy",
    "race_white":                   "Pct White",
    "median_age":                   "Median Age",
    "mortgage_rate_30yr":           "30yr Mortgage Rate",
    "mortgage_rate_30yr_momentum":  "Mortgage Rate Momentum",
    "treasury_10yr_yield":          "10yr Treasury Yield",
    "treasury_10yr_yield_momentum": "Treasury Yield Momentum",
    "fed_funds_rate":               "Fed Funds Rate",
    "fed_funds_rate_momentum":      "Fed Funds Rate Momentum",
    "local_unemployment_rate":      "Local Unemployment Rate",
    "local_unemployment_rate_momentum": "Unemployment Rate Momentum",
    "bw_sin":                       "Biweek Sin (Seasonality)",
    "bw_cos":                       "Biweek Cos (Seasonality)",
}

top15 = feat_df.head(15).copy()
top15["label"] = top15["feature"].map(FEATURE_LABELS).fillna(top15["feature"])

colors = sns.color_palette("flare", len(top15))[::-1]
bars = ax.barh(top15["label"][::-1], top15["mean_abs_attr"][::-1], color=colors)

ax.set_xlabel("Mean |GradientSHAP Attribution|", fontsize=12, fontweight="bold")
ax.set_title("LSTM Hazard Model — GradientSHAP Feature Attribution\n(2024 Out-of-Sample Test Set, Walk-Forward Fold)", 
             fontsize=14, fontweight="bold", pad=15)
ax.grid(axis="x", alpha=0.4)
sns.despine(left=True, bottom=False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "gradientshap_feature_importance.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved: {out_path}")
