import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

import ray
from ray import tune
from ray.train import Checkpoint

# PANEL_PATH will be uploaded to cluster via Ray
PANEL_PATH = "biweekly_panel.csv"

FEATS = [
    "proposed_max_height_ft", "existing_max_height_ft", "height_delta",
    "proposed_max_far", "existing_max_far",
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "lag1_cumulative_council_hearings",
    "commission_hearings_this_period", "lag1_cumulative_commission_hearings",
    "lag1_cumulative_petition_events", "lag1_cumulative_petition_count", "lag1_cumulative_petition_pct", 
    "lag1_Remand_Count",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta", 
    "fed_funds_rate", "fed_funds_rate_filing_delta", 
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
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
    for i, l in enumerate(lengths): mask[i, :l] = True
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
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out)

def train_lstm(config):
    # Retrieve dynamic params
    h_dim = config["hidden_dim"]
    lr = config["lr"]
    drop = config["dropout"]
    window = config["horizon_window"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prep data
    df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
    df_raw = df_raw.sort_values(["case_number", "period_seq"])
    df_raw["proposed_max_height_ft"] = df_raw["proposed_max_height_ft"].fillna(0)
    df_raw["existing_max_height_ft"] = df_raw["existing_max_height_ft"].fillna(0)
    df_raw["height_delta"] = df_raw["proposed_max_height_ft"] - df_raw["existing_max_height_ft"]
    
    df = df_raw.copy()
    if window == 1:
        df["target"] = df["petition_event"]
    else:
        df["target"] = df.groupby("case_number")["petition_event"].transform(
            lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].shift(-1)
        )
        df["target"] = df["target"].fillna(0)
    
    df["target"] = df["target"].astype(int)
    df = df[df["period_seq"] > 0]
    
    df[FEATS] = df[FEATS].fillna(0)
    scaler = StandardScaler()
    df[FEATS] = scaler.fit_transform(df[FEATS])
    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["case_number"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    
    train_dataset = ZoningHazardDataset(train_df, FEATS, "target")
    test_dataset = ZoningHazardDataset(test_df, FEATS, "target")
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
    
    pos_weight = torch.tensor([(len(train_df) - train_df["target"].sum()) / train_df["target"].sum()]).to(device)
    
    model = LSTMHazardModel(input_dim=len(FEATS), hidden_dim=h_dim, num_layers=2, dropout=drop).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(10):
        model.train()
        for batch_x, batch_y, lengths, mask in train_loader:
            batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, lengths)
            loss = criterion(logits[mask], batch_y[mask])
            loss.backward()
            optimizer.step()
            
    # Final Eval
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, lengths, mask in test_loader:
            batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
            logits = model(batch_x, lengths)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs[mask].cpu().numpy().flatten())
            all_trues.extend(batch_y[mask].cpu().numpy().flatten())
            
    pr_auc = average_precision_score(all_trues, all_preds)
    roc_auc = roc_auc_score(all_trues, all_preds)
    
    # Report back to Ray Tune Stream
    tune.report({"pr_auc": pr_auc, "roc_auc": roc_auc, "horizon": window})

if __name__ == "__main__":
    ray.init()
    
    # Define Grid Space
    search_space = {
        "hidden_dim": tune.grid_search([32, 64, 128, 256]),
        "lr": tune.grid_search([0.01, 0.005, 0.001]),
        "dropout": tune.grid_search([0.0, 0.1, 0.2]),
        "horizon_window": tune.grid_search([1, 6, 13, 26, 39, 52, 78, 104])
    }
    
    tuner = tune.Tuner(
        tune.with_resources(train_lstm, resources={"cpu": 2, "gpu": 0.25}),
        param_space=search_space,
    )
    
    results = tuner.fit()
    print("Best config: ", results.get_best_result(metric="pr_auc", mode="max").config)
