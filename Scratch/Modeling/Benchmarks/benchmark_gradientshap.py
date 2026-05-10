"""
GradientSHAP Benchmark — runs on the final 2024 fold's model.
Times attribution on N_SAMPLE cases from the test set.
"""
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from captum.attr import GradientShap

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
N_SAMPLE   = 25   # number of test cases to attribute
TRAIN_YEAR = 2024 # re-train on all data < 2024, attribute on 2024

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

# ── Quick re-train on final fold ─────────────────────────────────────────────
print(f"Re-training on data < {TRAIN_YEAR} ({len(train_df['case_number'].unique())} cases)...")
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
        out = model(bx, lens)
        loss = criterion(out[mask], by[mask])
        loss.backward(); optimizer.step()
        loss_sum += loss.item()
    print(f"  Epoch {epoch+1}/10 | Loss: {loss_sum/len(loader):.4f}", flush=True)

# ── Build fixed-length tensor for attribution ────────────────────────────────
model.eval()
test_groups = list(test_df.groupby("case_number"))[:N_SAMPLE]
MAX_LEN = max(len(g) for _, g in test_groups)

def pad_case(g):
    x = torch.tensor(g.sort_values("period_seq")[FEATS].values, dtype=torch.float32)
    pad = torch.zeros(MAX_LEN - len(x), len(FEATS))
    return torch.cat([x, pad], dim=0)

X_sample = torch.stack([pad_case(g) for _, g in test_groups]).to(device)  # [N, T, F]
lens_sample = torch.tensor([min(len(g), MAX_LEN) for _, g in test_groups])

# GradientSHAP requires a forward fn that returns a scalar per sample
# We use the final timestep logit
def forward_fn(x):
    # x: [N, T, F]
    out = model(x, lens_sample)  # [N, T, 1]
    # Return the max logit over time per sample (peak hazard)
    return out[:, -1, 0]  # last timestep

baseline = torch.zeros_like(X_sample)

print(f"\nRunning GradientSHAP on {N_SAMPLE} cases (max_len={MAX_LEN} periods, {len(FEATS)} features)...")

# cuDNN RNN requires training mode for gradient computation
model.train()

# wrap forward fn to keep model in train mode
def forward_fn(x):
    out = model(x, lens_sample)
    return out[:, -1, 0]

gs = GradientShap(forward_fn)

# Time first single case to extrapolate
print("Timing 1-case probe...", flush=True)
t_single = time.time()
_ = gs.attribute(X_sample[:1], baseline[:1], n_samples=50, stdevs=0.1)
elapsed_single = time.time() - t_single
n_test_total = len(test_df['case_number'].unique())
print(f"  1 case: {elapsed_single:.2f}s -> ",
      f"{N_SAMPLE} cases ~= {elapsed_single * N_SAMPLE:.1f}s, ",
      f"full test set ({n_test_total} cases) ~= {elapsed_single * n_test_total / 60:.1f} min", flush=True)

t0 = time.time()
attributions = gs.attribute(X_sample, baseline, n_samples=50, stdevs=0.1)
elapsed = time.time() - t0

print(f"\n✅ GradientSHAP complete!")
print(f"   Attribution shape: {attributions.shape}  [N_cases × T × Features]")
print(f"   Wall time: {elapsed:.2f}s for {N_SAMPLE} cases")
print(f"   Projected full test set (~{len(test_df['case_number'].unique())} cases): "
      f"{elapsed / N_SAMPLE * len(test_df['case_number'].unique()):.1f}s "
      f"({elapsed / N_SAMPLE * len(test_df['case_number'].unique()) / 60:.1f} min)")

# Top features by mean absolute attribution
mean_attr = attributions.abs().mean(dim=[0, 1]).cpu().detach().numpy()
feat_df = pd.DataFrame({"feature": FEATS, "mean_abs_attr": mean_attr})
feat_df = feat_df.sort_values("mean_abs_attr", ascending=False)
print("\nTop 10 Features by Mean |Attribution|:")
print(feat_df.head(10).to_string(index=False))
