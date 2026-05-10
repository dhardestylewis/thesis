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
from sklearn.model_selection import GroupShuffleSplit
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

horizons = {
    "14_Days": 1,
    "3_Months": 6,
    "6_Months": 13,
    "1_Year": 26,
    "1.5_Years": 39,
    "2_Years": 52,
    "3_Years": 78,
    "4_Years": 104
}

FEATS = [
    # Architectural Requests
    "proposed_max_height_ft", "existing_max_height_ft", "height_delta",
    "proposed_max_far", "existing_max_far",
    
    # Process & Bureaucracy
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "lag1_cumulative_council_hearings",
    "commission_hearings_this_period", "lag1_cumulative_commission_hearings",
    "lag1_cumulative_petition_events", "lag1_cumulative_petition_count", "lag1_cumulative_petition_pct", 
    "lag1_Remand_Count",
    
    # Economics & Demographics
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    
    # Macro Shocks
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta", 
    "fed_funds_rate", "fed_funds_rate_filing_delta", 
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    
    # Spatial Gravity
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "protest_density_100ft", "protest_density_150ft", "protest_density_200ft",
    "protest_density_250ft", "protest_density_300ft", "protest_density_350ft",
    "protest_density_400ft", "protest_density_450ft", "protest_density_500ft"
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

class GRUHazardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(GRUHazardModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        out, _ = self.gru(x)
        logits = self.fc(out)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results = []

for name, window in horizons.items():
    print(f"\n======================================", flush=True)
    print(f"Executing Horizon: {name} ({window} periods)", flush=True)
    
    df = df_raw.copy()
    
    if window == 1:
        df["target"] = df["petition_event"]
    else:
        df["target"] = df.groupby("case_number")["petition_event"].transform(
            lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].shift(-1)
        )
        df["target"] = df["target"].fillna(0)
    
    df["target"] = df["target"].astype(int)
    
    # Valid periods
    df = df[df["period_seq"] > 0]
    event_rate = df["target"].sum() / len(df)
    print(f"Event Rate: {event_rate*100:.2f}%", flush=True)
    
    # Scale
    df[FEATS] = df[FEATS].fillna(0)
    scaler = StandardScaler()
    df[FEATS] = scaler.fit_transform(df[FEATS])
    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["case_number"]))
    
    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]
    
    train_dataset = ZoningHazardDataset(train_df, FEATS, "target")
    test_dataset = ZoningHazardDataset(test_df, FEATS, "target")
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
    
    # Pos weight
    pos_weight = torch.tensor([(len(train_df) - train_df["target"].sum()) / train_df["target"].sum()]).to(device)
    
    model = GRUHazardModel(input_dim=len(FEATS), hidden_dim=64, num_layers=2).to(device)
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print("Torch compile failed or unavailable, using eager mode.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
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
        if (epoch + 1) % 5 == 0:
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
            
    pr_auc = average_precision_score(all_trues, all_preds)
    roc_auc = roc_auc_score(all_trues, all_preds)
    
    print(f"GRU PR AUC: {pr_auc:.4f} | ROC AUC: {roc_auc:.4f}", flush=True)
    
    results.append({
        "Horizon": name,
        "Window": window,
        "Event_Rate": event_rate,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    })

print("\nSaving GRU Gravity Curve...")
res_df = pd.DataFrame(results)
res_df.to_csv("gru_horizon_results.csv", index=False)

sns.set_theme(style="whitegrid", palette="mako")
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=res_df, x="Horizon", y="PR_AUC", marker="s", linewidth=3, ax=ax1, color="#005b96", label="GRU PR AUC")
ax1.set_ylabel("Precision-Recall AUC", fontsize=12, fontweight='bold', color="#005b96")
ax1.set_ylim(0, max(res_df["PR_AUC"].max() * 1.5, 0.1))

ax2 = ax1.twinx()
sns.barplot(data=res_df, x="Horizon", y="Event_Rate", alpha=0.3, ax=ax2, color="gray")
ax2.set_ylabel("Baseline Event Rate", fontsize=12, color="gray")
ax2.grid(False)

plt.title("Sequence Model (GRU) Performance Across Horizons", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\gru_gravity_curve.png", dpi=300, bbox_inches='tight')
print("Complete.")
