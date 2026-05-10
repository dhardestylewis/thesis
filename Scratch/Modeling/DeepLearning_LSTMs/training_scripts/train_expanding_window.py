import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
df_raw = df_raw.sort_values(["case_number", "period_seq"])

# Engineer Architectural Deltas
df_raw["proposed_max_height_ft"] = df_raw["proposed_max_height_ft"].fillna(0)
df_raw["existing_max_height_ft"] = df_raw["existing_max_height_ft"].fillna(0)
df_raw["height_delta"] = df_raw["proposed_max_height_ft"] - df_raw["existing_max_height_ft"]

# Temporal variables
df_raw["period_start"] = pd.to_datetime(df_raw["period_start"], errors="coerce")
df_raw = df_raw.dropna(subset=["period_start"])
df_raw["eval_year"] = df_raw["period_start"].dt.year

FEATS = [
    # Architectural Requests
    "proposed_max_height_ft", "existing_max_height_ft", "height_delta",
    "proposed_max_far", "existing_max_far",
    
    # Process & Bureaucracy
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count",
    
    # Economics & Demographics
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    
    # Macro Shocks
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", 
    "treasury_10yr_yield", "treasury_10yr_yield_momentum", 
    "fed_funds_rate", "fed_funds_rate_momentum", 
    "local_unemployment_rate", "local_unemployment_rate_momentum",
    
    # Spatial Gravity
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

class ZoningHazardDataset(Dataset):
    def __init__(self, data_df, features, target):
        self.groups = list(data_df.groupby("case_number"))
        self.features = features
        self.target = target
        
    def __len__(self):
        return len(self.groups)
        
    def __getitem__(self, idx):
        case_num, group = self.groups[idx]
        group = group.sort_values("period_seq")
        x = torch.tensor(group[self.features].values, dtype=torch.float32)
        y = torch.tensor(group[self.target].values, dtype=torch.float32).unsqueeze(1)
        return x, y

def collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    lengths = torch.tensor([len(x) for x in xs])
    
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    
    mask = torch.zeros(ys_padded.shape[:2], dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
        
    return xs_padded, ys_padded, lengths, mask

class LSTMHazardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMHazardModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x) # LSTM returns (out, (h_n, c_n))
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Focus exclusively on the 2-Year Horizon (52 periods)
window = 52
H_DIM = 256
LR = 0.001
DROP = 0.05

print(f"\n======================================", flush=True)
print(f"Executing 2-Year Horizon Expanding Window (Walk-Forward)", flush=True)
print(f"ARCHITECTURE: hidden_dim={H_DIM}, lr={LR}, dropout={DROP}", flush=True)

df = df_raw.copy()

# Target Engineer: 2 Year Window
df["target"] = df.groupby("case_number")["petition_event"].transform(
    lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].shift(-1)
)
df["target"] = df["target"].fillna(0)
df["target"] = df["target"].astype(int)

# Valid periods
df = df[df["period_seq"] > 0]

# Fill NaN — scaling done per-fold inside the loop to prevent distributional leakage
df[FEATS] = df[FEATS].fillna(0)

results = []
test_years = range(2016, 2025) # 2016 to 2024

for test_year in test_years:
    print(f"\n--- Fold: Test Year {test_year} ---", flush=True)
    
    # Expanding Window Split
    train_df = df[df["eval_year"] < test_year]
    test_df  = df[df["eval_year"] == test_year]
    
    if len(test_df) == 0:
        print(f"No test cases found for {test_year}. Skipping.", flush=True)
        continue

    # Fit scaler ONLY on train_df to prevent distributional leakage from future folds
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[FEATS] = scaler.fit_transform(train_df[FEATS])
    test_df[FEATS]  = scaler.transform(test_df[FEATS])
        
    train_event_rate = train_df["target"].sum() / len(train_df)
    test_event_rate = test_df["target"].sum() / len(test_df)
    
    print(f"Train Cases: {len(train_df['case_number'].unique())} | Test Cases: {len(test_df['case_number'].unique())}", flush=True)
    print(f"Train Event Rate: {train_event_rate*100:.2f}% | Test Event Rate: {test_event_rate*100:.2f}%", flush=True)
    
    train_dataset = ZoningHazardDataset(train_df, FEATS, "target")
    test_dataset = ZoningHazardDataset(test_df, FEATS, "target")
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
    
    # Pos weight dynamically adjusted per training window
    pos_weight = torch.tensor([(len(train_df) - train_df["target"].sum()) / max(train_df["target"].sum(), 1)]).to(device)
    
    model = LSTMHazardModel(input_dim=len(FEATS), hidden_dim=H_DIM, num_layers=2, dropout=DROP).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Train
    model.train()
    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y, lengths, mask in train_loader:
            batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, lengths)
            loss = criterion(logits[mask], batch_y[mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) == epochs:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}", flush=True)
    
    # Evaluate
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch_x, batch_y, lengths, mask in test_loader:
            batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
            logits = model(batch_x, lengths)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs[mask].cpu().numpy().flatten())
            all_trues.extend(batch_y[mask].cpu().numpy().flatten())
            
    # Calculate metrics
    if sum(all_trues) > 0:
        pr_auc = average_precision_score(all_trues, all_preds)
        roc_auc = roc_auc_score(all_trues, all_preds)
    else:
        pr_auc = 0.0
        roc_auc = 0.0
        print("Warning: No true positive events in test year.")
    
    print(f"Test Year {test_year} PR AUC: {pr_auc:.4f} | ROC AUC: {roc_auc:.4f}", flush=True)
    
    results.append({
        "Test_Year": test_year,
        "Train_Size_Parcels": len(train_df["case_number"].unique()),
        "Test_Size_Parcels": len(test_df["case_number"].unique()),
        "Test_Event_Rate": test_event_rate,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    })
    torch.save(model.state_dict(), f"fold_{test_year}_model.pt")
    print(f"  Model saved: fold_{test_year}_model.pt", flush=True)

print("\nSaving Walk-Forward Performance Curve...", flush=True)
res_df = pd.DataFrame(results)
res_df.to_csv("expanding_window_results.csv", index=False)

sns.set_theme(style="whitegrid", palette="flare")
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=res_df, x="Test_Year", y="PR_AUC", marker="o", linewidth=3, ax=ax1, color="#b00b69", label="Out-of-Sample PR AUC")
ax1.set_ylabel("Out-of-Sample PR AUC", fontsize=12, fontweight='bold', color="#b00b69")
ax1.set_ylim(0, max(res_df["PR_AUC"].max() * 1.5, 0.1))
ax1.set_xticks(res_df["Test_Year"])

ax2 = ax1.twinx()
sns.barplot(data=res_df, x="Test_Year", y="Test_Event_Rate", alpha=0.3, ax=ax2, color="gray")
ax2.set_ylabel("Test Set Baseline Event Rate", fontsize=12, color="gray")
ax2.grid(False)

plt.title("Expanding Window (Walk-Forward) Validation [2-Year Horizon]", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\expanding_window_performance.png", dpi=300, bbox_inches='tight')
print("Complete.", flush=True)
