"""
Spatial Attribution Map WITH BASEMAP — per-case GradientSHAP dominant driver.
Uses contextily CartoDB Dark Matter tiles over Austin metro.
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from captum.attr import GradientShap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import contextily as ctx

PANEL_PATH   = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
WEIGHTS_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\fold_2024_model.pt"
OUT_DIR      = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

# Austin metro bounds (WGS84)
LON_MIN, LON_MAX = -97.95, -97.55
LAT_MIN, LAT_MAX =  30.10,  30.50

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

CATEGORIES = {
    "proposed_max_height_ft": "Architectural", "existing_max_height_ft": "Architectural",
    "height_delta": "Architectural", "proposed_max_far": "Architectural",
    "existing_max_far": "Architectural",
    "period_seq": "Bureaucratic", "bw_sin": "Bureaucratic", "bw_cos": "Bureaucratic",
    "council_hearings_this_period": "Bureaucratic", "cumulative_council_hearings": "Bureaucratic",
    "commission_hearings_this_period": "Bureaucratic", "cumulative_commission_hearings": "Bureaucratic",
    "cumulative_petition_events": "Bureaucratic", "cumulative_petition_count": "Bureaucratic",
    "cumulative_petition_pct": "Bureaucratic", "Remand_Count": "Bureaucratic",
    "market_value": "Economic", "building_age": "Economic", "land_acres": "Economic",
    "total_population": "Economic", "median_household_income": "Economic",
    "renter_share": "Economic", "rent_burden": "Economic",
    "affordability_proxy": "Economic", "race_white": "Economic", "median_age": "Economic",
    "mortgage_rate_30yr": "Macro", "mortgage_rate_30yr_momentum": "Macro",
    "treasury_10yr_yield": "Macro", "treasury_10yr_yield_momentum": "Macro",
    "fed_funds_rate": "Macro", "fed_funds_rate_momentum": "Macro",
    "local_unemployment_rate": "Macro", "local_unemployment_rate_momentum": "Macro",
    "knn_petition_rate_1km": "Spatial Gravity", "dist_petition_rate_lag1": "Spatial Gravity",
}

CAT_COLORS = {
    "Architectural":  "#f97316",
    "Bureaucratic":   "#a78bfa",
    "Economic":       "#34d399",
    "Macro":          "#60a5fa",
    "Spatial Gravity":"#f43f5e",
}

def to_mercator(lon, lat):
    """Convert WGS84 lon/lat arrays to Web Mercator (EPSG:3857)."""
    import math
    R = 6378137.0
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi/4 + np.radians(lat)/2)) * R
    return x, y

# Precompute bounding box in Mercator
x_min, y_min = to_mercator(LON_MIN, LAT_MIN)
x_max, y_max = to_mercator(LON_MAX, LAT_MAX)

device = torch.device("cpu")
print("Device: CPU")

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
model.train()
print("Weights loaded.")

test_groups = list(test_df.groupby("case_number"))
MAX_LEN = max(len(g) for _,g in test_groups)

def pad_case(g):
    x = torch.tensor(g.sort_values("period_seq")[FEATS].values, dtype=torch.float32)
    return torch.cat([x, torch.zeros(MAX_LEN - len(x), len(FEATS))], dim=0)

X    = torch.stack([pad_case(g) for _,g in test_groups])
lens = torch.tensor([min(len(g), MAX_LEN) for _,g in test_groups])

def fwd(x): return model(x, lens)[:, -1, 0]
gs = GradientShap(fwd)
print("Running GradientSHAP...", flush=True)
attrs = gs.attribute(X, torch.zeros_like(X), n_samples=50, stdevs=0.1)
print("Done.")

per_case_attr = attrs.abs().mean(dim=1).detach().numpy()

model.eval()
with torch.no_grad():
    peak_prob = torch.sigmoid(model(X, lens)).squeeze(-1).max(dim=1).values.numpy()

actuals   = [g["target"].max() for _, g in test_groups]
case_nums = [cn for cn, _ in test_groups]

dominant_feat_idx = per_case_attr.argmax(axis=1)
dominant_cat = [CATEGORIES[FEATS[i]] for i in dominant_feat_idx]

cat_names = list(CAT_COLORS.keys())
feat_to_cat_idx = {f: cat_names.index(CATEGORIES[f]) for f in FEATS}
cat_attr = np.zeros((len(test_groups), len(cat_names)))
for fi, f in enumerate(FEATS):
    cat_attr[:, feat_to_cat_idx[f]] += per_case_attr[:, fi]
total = cat_attr.sum(axis=1, keepdims=True)
total[total == 0] = 1
cat_attr_pct = cat_attr / total * 100

coords = (
    test_df.sort_values("period_seq")
    .groupby("case_number")[["latitude","longitude","council_district"]]
    .last().reset_index()
)

result_df = pd.DataFrame({
    "case_number":  case_nums,
    "peak_prob":    peak_prob,
    "actual_event": actuals,
    "dominant_cat": dominant_cat,
}).merge(coords, on="case_number", how="left")
result_df = result_df.dropna(subset=["latitude","longitude"])

for i, cat in enumerate(cat_names):
    result_df[f"pct_{cat}"] = cat_attr_pct[:len(result_df), i]

# Clip to Austin core + project to Mercator
df_plot = result_df[
    result_df["longitude"].between(LON_MIN, LON_MAX) &
    result_df["latitude"].between(LAT_MIN, LAT_MAX)
].copy()
df_plot["mx"], df_plot["my"] = to_mercator(df_plot["longitude"].values, df_plot["latitude"].values)
ev_plot = df_plot[df_plot["actual_event"] == 1]

TILE = ctx.providers.CartoDB.DarkMatter

# ── Plot 1: Dominant Driver Map ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
ax.set_facecolor("#0d1117")

for cat, color in CAT_COLORS.items():
    sub = df_plot[df_plot["dominant_cat"] == cat]
    if len(sub) == 0: continue
    ax.scatter(sub["mx"], sub["my"], c=color,
               s=(sub["peak_prob"]*120+15).values,
               alpha=0.8, edgecolors="none", zorder=3, label=cat)

if len(ev_plot) > 0:
    ax.scatter(ev_plot["mx"], ev_plot["my"],
               c=[CAT_COLORS[c] for c in ev_plot["dominant_cat"]],
               s=(ev_plot["peak_prob"]*150+30).values,
               edgecolors="white", linewidths=1.5, zorder=4)

ctx.add_basemap(ax, source=TILE, zoom=12, alpha=0.85)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_axis_off()

handles = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
handles.append(plt.scatter([], [], edgecolors="white", facecolors="gray",
                           linewidths=1.5, s=80, label="Confirmed Petition"))
ax.legend(handles=handles, facecolor="#1a1a2e", edgecolor="#555",
          labelcolor="white", fontsize=10, loc="upper left")

ax.set_title(
    "Dominant Attribution Driver — Spatial Distribution\n"
    "Dynamic LSTM | 2024 Out-of-Sample Fold | Point Size = Predicted Probability",
    color="white", fontsize=13, fontweight="bold", pad=10
)
fig.patch.set_facecolor("#0d1117")
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "spatial_attribution_dominant_basemap_2024.png")
plt.savefig(p1, dpi=300, bbox_inches="tight", facecolor="#0d1117")
print(f"Dominant driver basemap saved: {p1}")

# ── Plot 2: Faceted — top 3 categories ───────────────────────────────────────
top3_cats = ["Architectural", "Spatial Gravity", "Bureaucratic"]
fig, axes = plt.subplots(1, 3, figsize=(21, 8))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(
    "Per-Case Category Attribution Intensity — Spatial Distribution (2024 Fold)",
    color="white", fontsize=13, fontweight="bold", y=1.01
)

for ax, cat in zip(axes, top3_cats):
    col = f"pct_{cat}"
    norm = mcolors.Normalize(vmin=0, vmax=max(df_plot[col].clip(lower=0).quantile(0.95), 1e-6))
    sc = ax.scatter(df_plot["mx"], df_plot["my"],
                    c=df_plot[col].clip(lower=0), cmap="YlOrRd", norm=norm,
                    s=35, alpha=0.85, edgecolors="none", zorder=3)
    if len(ev_plot) > 0:
        ax.scatter(ev_plot["mx"], ev_plot["my"],
                   c=ev_plot[col].clip(lower=0), cmap="YlOrRd", norm=norm,
                   s=140, edgecolors="white", linewidths=1.3, zorder=4)
    ctx.add_basemap(ax, source=TILE, zoom=12, alpha=0.85)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, shrink=0.8)
    cbar.set_label("% of Attribution", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title(cat, color=CAT_COLORS[cat], fontsize=13, fontweight="bold", pad=8)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "spatial_attribution_faceted_basemap_2024.png")
plt.savefig(p2, dpi=300, bbox_inches="tight", facecolor="#0d1117")
print(f"Faceted basemap saved: {p2}")
