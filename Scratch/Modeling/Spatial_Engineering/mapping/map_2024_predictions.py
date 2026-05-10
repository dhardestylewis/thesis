"""
Spatial risk map for the 2024 Walk-Forward test fold.
Loads fold_2024_model.pt, runs CPU inference on 2024 test cases,
plots predicted protest probability on an Austin map.
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

PANEL_PATH   = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
WEIGHTS_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\fold_2024_model.pt"
OUT_DIR      = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

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

device = torch.device("cpu")  # CPU only
print("Device: CPU")

# ── Load panel ──────────────────────────────────────────────────────────────
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

test_df = df_raw[df_raw["eval_year"] == 2024].copy()
print(f"2024 test cases: {test_df['case_number'].nunique()}")

# ── Model ────────────────────────────────────────────────────────────────────
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

model = LSTMHazardModel(input_dim=len(FEATS))
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()
print("Weights loaded.")

# ── Inference: get peak hazard probability per case ──────────────────────────
test_groups = list(test_df.groupby("case_number"))
MAX_LEN = max(len(g) for _,g in test_groups)

def pad_case(g):
    x = torch.tensor(g.sort_values("period_seq")[FEATS].values, dtype=torch.float32)
    return torch.cat([x, torch.zeros(MAX_LEN - len(x), len(FEATS))], dim=0)

X = torch.stack([pad_case(g) for _,g in test_groups])
lens = torch.tensor([min(len(g), MAX_LEN) for _,g in test_groups])

print("Running inference on CPU...")
with torch.no_grad():
    logits = model(X, lens)          # [N, T, 1]
    probs  = torch.sigmoid(logits)   # [N, T, 1]
    peak_prob = probs.squeeze(-1).max(dim=1).values.numpy()  # [N] peak over sequence

case_nums = [cn for cn, _ in test_groups]
actuals   = [g["target"].max() for _, g in test_groups]

# Pull lat/lon from the last observed period per case
coords = (
    test_df.sort_values("period_seq")
    .groupby("case_number")[["latitude","longitude"]]
    .last()
    .reset_index()
)

result_df = pd.DataFrame({
    "case_number": case_nums,
    "peak_prob": peak_prob,
    "actual_event": actuals
}).merge(coords, on="case_number", how="left")

result_df = result_df.dropna(subset=["latitude","longitude"])
print(f"Cases with coordinates: {len(result_df)}")

# Save predictions
result_df.to_csv(os.path.join(OUT_DIR, "spatial_predictions_2024.csv"), index=False)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

# Background scatter — all cases
norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = cm.get_cmap("plasma")

non_events = result_df[result_df["actual_event"] == 0]
events     = result_df[result_df["actual_event"] == 1]

# Plot non-events (smaller, dimmer)
sc = ax.scatter(
    non_events["longitude"], non_events["latitude"],
    c=non_events["peak_prob"], cmap="plasma", norm=norm,
    s=30, alpha=0.6, edgecolors="none", zorder=2
)

# Plot actual petition events (larger, with ring)
ax.scatter(
    events["longitude"], events["latitude"],
    c=events["peak_prob"], cmap="plasma", norm=norm,
    s=120, alpha=1.0, edgecolors="white", linewidths=1.2, zorder=3,
    label="Confirmed Protest Petition"
)

cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Predicted Protest Probability", color="white", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

ax.set_xlabel("Longitude", color="white", fontsize=10)
ax.set_ylabel("Latitude", color="white", fontsize=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

ax.set_title(
    "Dynamic LSTM — 2-Year Protest Hazard\nSpatial Risk Map | 2024 Out-of-Sample Test Fold",
    color="white", fontsize=14, fontweight="bold", pad=15
)
ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "spatial_risk_map_2024.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#0d1117")
print(f"\nSpatial map saved: {out_path}")
print(f"  Cases plotted: {len(result_df)}")
print(f"  Actual events: {events.shape[0]}")
print(f"  Mean predicted prob: {result_df['peak_prob'].mean():.3f}")
print(f"  Max predicted prob:  {result_df['peak_prob'].max():.3f}")
