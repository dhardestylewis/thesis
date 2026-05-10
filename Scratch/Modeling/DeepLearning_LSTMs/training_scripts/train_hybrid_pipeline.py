import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
import sys

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "petition_event"
HIDDEN_DIM = 64

print("1. Loading Bi-Weekly Panel...")
df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
df = df_raw[df_raw["period_seq"] > 0].copy()

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

print("2. Preprocessing & Tensorization for GRU...")
df[FEATS] = df[FEATS].fillna(0)
scaler = StandardScaler()
df[FEATS] = scaler.fit_transform(df[FEATS])

# Crucial: retain the original raw_period_seq for safe merging later
df["raw_period_seq"] = df_raw["period_seq"]

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

class HazardGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.fc(out)
        return logits, out # return the intermediate hidden states 'out'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HazardGRU(input_dim=len(FEATS)).to(device)

pos_events = train_df[TARGET].sum()
neg_events = len(train_df) - pos_events
pos_weight = torch.tensor([neg_events / pos_events]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print("\n3. Training GRU to Learn Sequence Latents...")
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

print("\n4. Extracting 64-Dimensional Latent Sequence Embeddings...")
all_dataset = ZoningHazardDataset(df)
all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

latent_records = []

model.eval()
with torch.no_grad():
    for batch_idx, (x, y, mask) in enumerate(all_loader):
        x = x.to(device)
        logits, latents = model(x) # latents shape: (batch, seq, 64)
        latents_np = latents.cpu().numpy()
        
        start_idx = batch_idx * 64
        for i in range(len(x)):
            case_idx = start_idx + i
            if case_idx >= len(all_dataset.groups): break
            case_num, group = all_dataset.groups[case_idx]
            group = group.sort_values("period_seq")
            
            seq_len = int(mask[i].sum().item())
            
            for t in range(seq_len):
                period = group.iloc[t]["raw_period_seq"]
                hidden_state = latents_np[i, t, :]
                
                record = {
                    "case_number": case_num,
                    "period_seq": period
                }
                # Add the 64 embedding dimensions
                for d in range(HIDDEN_DIM):
                    record[f"gru_embed_{d}"] = hidden_state[d]
                    
                latent_records.append(record)

latents_df = pd.DataFrame(latent_records)

print("\n5. Merging Latents with Raw Dataset for Hybrid Training...")
# Load the completely raw panel so CatBoost gets the unscaled features!
hybrid_df = pd.read_csv(PANEL_PATH, low_memory=False)
hybrid_df = hybrid_df[hybrid_df["period_seq"] > 0].copy()
hybrid_df["period_seq"] = hybrid_df["period_seq"].astype(int)
latents_df["period_seq"] = latents_df["period_seq"].astype(int)

hybrid_df = hybrid_df.merge(latents_df, on=["case_number", "period_seq"], how="inner")

LATENT_FEATS = [f"gru_embed_{d}" for d in range(HIDDEN_DIM)]
ALL_FEATS = FEATS + LATENT_FEATS

hybrid_df[ALL_FEATS] = hybrid_df[ALL_FEATS].fillna(0)

print(f"Hybrid Feature Set: {len(FEATS)} Tabular + {len(LATENT_FEATS)} Neural = {len(ALL_FEATS)} Total")

# Split using the SAME split logic so we don't leak
train = hybrid_df[hybrid_df["case_number"].isin(train_df["case_number"])]
test  = hybrid_df[hybrid_df["case_number"].isin(test_df["case_number"])]

X_train, y_train = train[ALL_FEATS], train[TARGET]
X_test, y_test   = test[ALL_FEATS], test[TARGET]

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

print("\n6. Training Hybrid CatBoost (Tabular + Neural Embeddings)...")
clf = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='AUC',
    random_seed=42,
    verbose=False,
    task_type="GPU"
)

clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    verbose=100
)

print("\n7. Final Hybrid Evaluation...")
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print(f"   ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))
