import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "vote_event"
print(f"TARGET: {TARGET}")

# Load data
print("1. Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)
df = df[df["period_seq"] > 0].copy()

# Features
FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "cumulative_min_signer_dist", "cumulative_max_signer_dist", "cumulative_median_signer_dist", 
    "cumulative_signers_within_200ft", "cumulative_signers_outside_200ft", 
    "cumulative_unofficial_protest_intensity", 
    "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2", "cumulative_protester_embed_dim3", "cumulative_protester_embed_dim4",
    "cumulative_temporal_protesting_pct_sf", "cumulative_temporal_silent_pct_sf",
    "cumulative_temporal_protesting_pct_com", "cumulative_temporal_silent_pct_com",
    "cumulative_temporal_protesting_pct_mf", "cumulative_temporal_silent_pct_mf",
    "cumulative_delta_protesting_friction", "cumulative_delta_silent_friction",
    "Remand_Count",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta", 
    "fed_funds_rate", "fed_funds_rate_filing_delta", 
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

print("2. Preprocessing & Tensorization...")
# Impute and Scale (Crucial for Neural Networks)
df[FEATS] = df[FEATS].fillna(0)
scaler = StandardScaler()
df[FEATS] = scaler.fit_transform(df[FEATS])

# Group split to avoid leakage
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["case_number"]))

train_df = df.iloc[train_idx]
test_df  = df.iloc[test_idx]

class ZoningHazardDataset(Dataset):
    def __init__(self, data_df):
        self.groups = list(data_df.groupby("case_number"))
        
    def __len__(self):
        return len(self.groups)
        
    def __getitem__(self, idx):
        case_num, group = self.groups[idx]
        group = group.sort_values("period_seq")
        x = torch.tensor(group[FEATS].values, dtype=torch.float32)
        y = torch.tensor(group[TARGET].values, dtype=torch.float32).unsqueeze(1)
        return x, y

def collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    lengths = torch.tensor([len(x) for x in xs])
    
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0.0)
    
    mask = torch.zeros(ys_padded.shape, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l, 0] = True
        
    return xs_padded, ys_padded, mask

train_dataset = ZoningHazardDataset(train_df)
test_dataset = ZoningHazardDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

print(f"   Train cases: {len(train_dataset)} | Test cases: {len(test_dataset)}")

class HazardLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out # logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HazardLSTM(input_dim=len(FEATS)).to(device)

# Calculate pos_weight for BCEWithLogitsLoss
pos_events = train_df[TARGET].sum()
neg_events = len(train_df) - pos_events
pos_weight = torch.tensor([neg_events / pos_events]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print("\n3. Training LSTM...")
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Apply mask to only calculate loss on valid timesteps
        loss = criterion(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

print("\n4. Evaluating LSTM...")
model.eval()
all_preds = []
all_trues = []

with torch.no_grad():
    for x, y, mask in test_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        
        all_preds.extend(probs[mask].cpu().numpy().flatten())
        all_trues.extend(y[mask].cpu().numpy().flatten())

y_true = np.array(all_trues)
y_pred_proba = np.array(all_preds)
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"\n   ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y_true, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_true, y_pred))
