"""
time_benchmark.py
Single-cutoff, single-horizon timing probe.
Runs each model once, reports per-model wall time,
then extrapolates to the full 08e grid.
"""
import time
import gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT       = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv"

FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
    "cumulative_min_signer_dist", "cumulative_max_signer_dist", "cumulative_median_signer_dist",
    "cumulative_signers_within_200ft", "cumulative_signers_outside_200ft",
    "cumulative_unofficial_protest_intensity",
    "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2",
    "cumulative_protester_embed_dim3", "cumulative_protester_embed_dim4",
    "cumulative_temporal_protesting_pct_sf", "cumulative_temporal_silent_pct_sf",
    "cumulative_temporal_protesting_pct_com", "cumulative_temporal_silent_pct_com",
    "cumulative_temporal_protesting_pct_mf", "cumulative_temporal_silent_pct_mf",
    "cumulative_delta_protesting_friction", "cumulative_delta_silent_friction",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income",
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta",
    "fed_funds_rate", "fed_funds_rate_filing_delta",
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "active_cases_100m", "active_cases_250m", "active_cases_500m",
    "active_cases_1km", "active_cases_2km", "active_gravity_index_t",
    "hearing_frequency", "petition_intensity_per_ft",
    "hearing_velocity_3p", "petition_velocity_3p",
    "pdf_requested_height_ft", "pdf_requested_max_far", "pdf_proposed_height_ft",
    "pdf_story_count", "pdf_story_height_ft", "pdf_compatibility_height_ft",
]

# ── Grid constants (matching 08e) ─────────────────────────────────────────────
# 9 cutoffs × 5 horizons, but ~40% skipped (no positives in early years)
TOTAL_CUTOFFS  = 9
TOTAL_HORIZONS = 5
SKIP_RATE      = 0.40   # fraction of (cutoff, horizon) cells expected to be skipped
ACTIVE_CELLS   = round(TOTAL_CUTOFFS * TOTAL_HORIZONS * (1 - SKIP_RATE))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")
print(f"Active cells estimate: {ACTIVE_CELLS} / {TOTAL_CUTOFFS * TOTAL_HORIZONS}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading panel...")
t0 = time.perf_counter()
df = pd.read_csv(PANEL_PATH, low_memory=False)
df = df.sort_values(["case_number", "period_seq"]).reset_index(drop=True)
load_time = time.perf_counter() - t0
print(f"  Loaded in {load_time:.1f}s — {len(df):,} rows")

feats = [f for f in FEATS if f in df.columns]
print(f"  {len(feats)} features available")

# Use 2022 as probe cutoff — large enough train set, has positives
PROBE_CUTOFF = 2022
train_mask = df["year"].values < PROBE_CUTOFF
test_mask  = df["year"].values == PROBE_CUTOFF

# 1-year horizon target (window=26 biweekly periods)
target = df.groupby("case_number")["petition_event"].transform(
    lambda x: x.iloc[::-1].rolling(window=26, min_periods=1).max().iloc[::-1]
).fillna(0).astype(int).values

X_tr = df[feats].fillna(0).values[train_mask]
X_te = df[feats].fillna(0).values[test_mask]
y_tr = target[train_mask]
y_te = target[test_mask]

df_lstm_tr = df[["case_number"] + feats].iloc[np.where(train_mask)[0]].reset_index(drop=True)
df_lstm_te = df[["case_number"] + feats].iloc[np.where(test_mask)[0]].reset_index(drop=True)

print(f"\n  Train: {len(X_tr):,} rows | {y_tr.sum()} positives ({y_tr.mean():.1%})")
print(f"  Test : {len(X_te):,} rows | {y_te.sum()} positives ({y_te.mean():.1%})")

scaler = StandardScaler().fit(X_tr)
spw    = max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))

timings = {}

# ── CatBoost ──────────────────────────────────────────────────────────────────
print("\n--- CatBoost ---")
try:
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="AUC",
        random_seed=42, verbose=False,
        task_type="GPU" if DEVICE == "cuda" else "CPU",
        thread_count=1 if DEVICE == "cuda" else -1,
    )
    t0 = time.perf_counter()
    cb.fit(X_tr, y_tr, verbose=False)
    cb.predict_proba(X_te)
    timings["CatBoost"] = time.perf_counter() - t0
    print(f"  {timings['CatBoost']:.1f}s")
except ImportError:
    print("  SKIP — catboost not installed in this venv")
    timings["CatBoost"] = None

# ── XGB-RF ────────────────────────────────────────────────────────────────────
print("--- XGB-RF (150 trees) ---")
try:
    from xgboost import XGBRFClassifier
    rf = XGBRFClassifier(
        n_estimators=150, max_depth=8, scale_pos_weight=spw,
        tree_method="hist", device=DEVICE,
        n_jobs=1 if DEVICE == "cuda" else -1, random_state=42,
    )
    t0 = time.perf_counter()
    rf.fit(X_tr, y_tr)
    rf.predict_proba(X_te)
    timings["RandomForest"] = time.perf_counter() - t0
    print(f"  {timings['RandomForest']:.1f}s")
except ImportError:
    print("  SKIP — xgboost not installed in this venv")
    timings["RandomForest"] = None

# ── sklearn Logistic L2 ───────────────────────────────────────────────────────
print("--- LogisticL2 (sklearn lbfgs) ---")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
lr2 = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(C=1.0, solver="lbfgs", penalty="l2",
                               max_iter=500, tol=1e-3, n_jobs=-1,
                               class_weight="balanced")),
])
t0 = time.perf_counter()
lr2.fit(X_tr, y_tr)
lr2.predict_proba(X_te)
timings["LogisticL2"] = time.perf_counter() - t0
print(f"  {timings['LogisticL2']:.1f}s")

# ── sklearn Logistic L1 ───────────────────────────────────────────────────────
print("--- LogisticL1 (sklearn saga) ---")
lr1 = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(C=1.0, solver="saga", penalty="l1",
                               max_iter=500, tol=1e-3,
                               class_weight="balanced")),
])
t0 = time.perf_counter()
lr1.fit(X_tr, y_tr)
lr1.predict_proba(X_te)
timings["LogisticL1"] = time.perf_counter() - t0
print(f"  {timings['LogisticL1']:.1f}s")

# ── sklearn Linear ────────────────────────────────────────────────────────────
print("--- Linear (sklearn lbfgs C=100) ---")
lin = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(C=100.0, solver="lbfgs", penalty="l2",
                               max_iter=500, tol=1e-3, n_jobs=-1,
                               class_weight="balanced")),
])
t0 = time.perf_counter()
lin.fit(X_tr, y_tr)
lin.predict_proba(X_te)
timings["Linear"] = time.perf_counter() - t0
print(f"  {timings['Linear']:.1f}s")

# ── PyTorch MLP ───────────────────────────────────────────────────────────────
print("--- MLP (128->64->32, 30 epochs) ---")
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def train_mlp(X_tr, y_tr, X_te, scaler, hidden_dims, epochs, lr=1e-3, batch_size=2048):
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers, d = [], X_tr.shape[1]
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers.append(nn.Linear(d, 1))
    model = nn.Sequential(*layers).to(device)
    pw    = max(1.0, (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(device))
    opt   = optim.Adam(model.parameters(), lr=lr)
    Xt    = torch.tensor(X_tr_sc, dtype=torch.float32)
    yt    = torch.tensor(y_tr,    dtype=torch.float32).unsqueeze(1)
    dl    = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(Xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_te_sc, dtype=torch.float32).to(device))).cpu().numpy().flatten()
    del Xt, yt, dl, model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return probs

t0 = time.perf_counter()
train_mlp(X_tr, y_tr, X_te, scaler, hidden_dims=[128, 64, 32], epochs=30)
timings["MLP"] = time.perf_counter() - t0
print(f"  {timings['MLP']:.1f}s")

# ── PyTorch LSTM ──────────────────────────────────────────────────────────────
print("--- LSTM (64 hidden, 10 epochs) ---")
def make_3d(X_vals, cases, seq_len=6):
    n = len(X_vals)
    X_3d = np.zeros((n, seq_len, X_vals.shape[1]), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        while start < i and cases[start] != cases[i]:
            start += 1
        vlen = i - start + 1
        X_3d[i, -vlen:, :] = X_vals[start:i + 1, :]
    return torch.tensor(X_3d)

class SimpleLSTM(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.fc   = nn.Linear(h, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

t0 = time.perf_counter()
Xtr_sc = scaler.transform(df_lstm_tr[feats].fillna(0).values)
Xte_sc = scaler.transform(df_lstm_te[feats].fillna(0).values)
X3d_tr = make_3d(Xtr_sc, df_lstm_tr["case_number"].values)
X3d_te = make_3d(Xte_sc, df_lstm_te["case_number"].values)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SimpleLSTM(len(feats), 64).to(device)
pw     = max(1.0, (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
crit   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(device))
opt    = optim.Adam(model.parameters(), lr=0.01)
dl     = DataLoader(TensorDataset(X3d_tr, y_tr_t), batch_size=2048, shuffle=True)
model.train()
for _ in range(10):
    for Xb, yb in dl:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad(); crit(model(Xb), yb).backward(); opt.step()
model.eval()
with torch.no_grad():
    torch.sigmoid(model(X3d_te.to(device))).cpu().numpy()
timings["LSTM"] = time.perf_counter() - t0
print(f"  {timings['LSTM']:.1f}s")

# ── Summary + Extrapolation ───────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  {'Model':<16} {'Per cell':>10}   {'× '+str(ACTIVE_CELLS)+' cells':>12}")
print("-"*60)
total_sec = 0
for m, t in timings.items():
    if t is None:
        print(f"  {m:<16} {'(skipped)':>10}   {'N/A':>12}")
        continue
    projected = t * ACTIVE_CELLS
    total_sec += projected
    print(f"  {m:<16} {t:>9.1f}s   {projected/60:>10.1f} min")
print("-"*60)
print(f"  {'TOTAL (excl skip)':<16}             {total_sec/60:>9.1f} min")
print(f"  {'+ boot/pip':<16}             {'~5':>9} min")
print(f"  {'WALL CLOCK EST':<16}             {(total_sec+300)/60:>9.1f} min")
print("="*60)
print(f"\nDevice used: {DEVICE}")
print(f"On g4dn.xlarge T4 GPU: CatBoost/RF ~5-8x faster than CPU baseline.")
