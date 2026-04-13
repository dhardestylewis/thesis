"""
irm_vrex_nonlinear.py — PyTorch Representation Learning for Invariance
======================================================================
Tests if an MLP feature extractor (Phi) can learn an invariant representation
across the 115 Austin zoning environments (2018-2022 train), such that a 
fixed linear classifier w=1.0 generalizes to 2023-2025.

Models:
  1. ERM   — Standard supervised MLP
  2. IRM   — MLP + IRMv1 gradient penalty
  3. V-REx — MLP + Variance(Risk^e) penalty

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "Analysis", "Results")

FEATURE_COLS = [
    'year_built', 'deed_acreage',
    'land_market_value', 'year',
    'land_acres', 'new_construction_value'
]
TARGET_COL = 'total_market_value'
MIN_ENV_SIZE = 5

# --- PyTorch Models ---

class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)  # Output scalar target directly
        )

    def forward(self, x):
        return self.net(x)

# --- Training Routines ---

def train_irm(model, optimizer, train_loader, n_envs, epochs=100, lam=100.0, method="IRM"):
    """
    Trains the model using ERM, IRMv1, or V-REx penalties.
    method can be "ERM", "IRM", or "V-REx"
    """
    model.train()
    criterion = nn.MSELoss(reduction='none')
    
    # Needs a dummy scale parameter w=1.0 for IRMv1 gradient penalty
    dummy_w = torch.tensor(1.0, requires_grad=True, device=next(model.parameters()).device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for x_batch, y_batch, env_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            phi = model(x_batch).squeeze(-1)
            
            # ERM Loss
            loss = criterion(phi * dummy_w, y_batch)
            
            # Compute risks per environment
            unique_envs = torch.unique(env_batch)
            env_risks = []
            irm_penalties = []
            
            for e in unique_envs:
                mask = env_batch == e
                if mask.sum() < 2: continue
                
                env_loss = loss[mask].mean()
                env_risks.append(env_loss)
                
                if method == "IRM":
                    # Gradient of environment risk with respect to dummy w
                    grad = torch.autograd.grad(env_loss, [dummy_w], create_graph=True)[0]
                    irm_penalties.append(grad ** 2)
            
            if not env_risks:
                continue
                
            env_risks_stack = torch.stack(env_risks)
            erm_loss = env_risks_stack.mean()
            
            if method == "ERM":
                final_loss = erm_loss
            elif method == "IRM":
                penalty = torch.stack(irm_penalties).mean()
                # Scale lambda up linearly over the first 20 epochs
                lambda_weight = lam if epoch > 20 else lam * (epoch / 20.0)
                final_loss = erm_loss + lambda_weight * penalty
            elif method == "V-REx":
                penalty = env_risks_stack.var()
                beta_weight = lam if epoch > 20 else lam * (epoch / 20.0)
                final_loss = erm_loss + beta_weight * penalty
            
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            total_loss += final_loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d} | {method} Loss: {total_loss/len(train_loader):.4f}")


@torch.no_grad()
def evaluate_risks(model, x, y, envs):
    model.eval()
    phi = model(x).squeeze(-1)
    criterion = nn.MSELoss(reduction='none')
    losses = criterion(phi, y)
    
    unique_envs = torch.unique(envs)
    risks = []
    
    for e in unique_envs:
        mask = envs == e
        if mask.sum() >= 2:
            risks.append(losses[mask].mean().item())
            
    risks = np.array(risks)
    return risks.mean(), risks.max(), risks.max() - risks.min(), risks.std()


def load_dataset():
    print("Loading environment assignments...")
    env = pd.read_csv(ENV_PATH)
    env = env.rename(columns={'CASE_NUMBER': 'env_id', 'SUB_TYPE': 'env_type'})
    
    print("Loading panel (selective columns)...")
    usecols = ['standardized_tcad_id', 'year'] + FEATURE_COLS + [TARGET_COL]
    panel = pd.read_csv(PANEL_PATH, usecols=usecols, low_memory=False)
    panel = panel[panel['year'].between(2018, 2025)]
    
    df = panel.merge(env, on='standardized_tcad_id', how='inner')
    df = df[df[TARGET_COL] > 0].copy()
    df['log_tmv'] = np.log(df[TARGET_COL])
    
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['new_construction_value'] = df['new_construction_value'].fillna(0)
    df = df.dropna(subset=FEATURE_COLS + ['log_tmv'])
    
    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]
    
    # Map string env_ids to integer labels for PyTorch
    env_map = {name: i for i, name in enumerate(df['env_id'].unique())}
    df['env_label'] = df['env_id'].map(env_map)
    
    return df

def main():
    df = load_dataset()
    
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['log_tmv'].values.astype(np.float32)
    envs = df['env_label'].values.astype(np.int64)
    years = df['year'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    train_mask = years <= 2022
    test_mask = years >= 2023
    
    X_train, y_train, env_train = X[train_mask], y[train_mask], envs[train_mask]
    X_test, y_test, env_test = X[test_mask], y[test_mask], envs[test_mask]
    
    # Ensure Test envs overlap Train envs
    common_envs = set(np.unique(env_train)) & set(np.unique(env_test))
    train_keep = np.isin(env_train, list(common_envs))
    test_keep = np.isin(env_test, list(common_envs))
    
    X_train, y_train, env_train = X_train[train_keep], y_train[train_keep], env_train[train_keep]
    X_test, y_test, env_test = X_test[test_keep], y_test[test_keep], env_test[test_keep]

    print(f"\nTrain: {len(y_train):,} obs, {len(np.unique(env_train))} envs (2018-2022)")
    print(f"Test:  {len(y_test):,} obs, {len(np.unique(env_test))} envs (2023-2025)")

    # To PyTorch Tensors
    X_tr_t = torch.tensor(X_train)
    y_tr_t = torch.tensor(y_train)
    env_tr_t = torch.tensor(env_train)
    
    X_te_t = torch.tensor(X_test)
    y_te_t = torch.tensor(y_test)
    env_te_t = torch.tensor(env_test)

    train_ds = TensorDataset(X_tr_t, y_tr_t, env_tr_t)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    
    results = []

    for method, lam in [("ERM", 0.0), ("IRM", 10.0), ("V-REx", 10.0)]:
        print("\n" + "=" * 60)
        print(f"Training {method} MLP")
        print("=" * 60)
        
        model = MLPFeatureExtractor(input_dim=X.shape[1], hidden_dim=64)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        train_irm(model, optimizer, train_loader, len(common_envs), epochs=300, lam=lam, method=method)
        
        # Evaluate
        tr_mean, tr_worst, tr_gap, tr_std = evaluate_risks(model, X_tr_t, y_tr_t, env_tr_t)
        te_mean, te_worst, te_gap, te_std = evaluate_risks(model, X_te_t, y_te_t, env_te_t)
        
        results.append({
            'Method': method,
            'Train Mean': tr_mean, 'Test Mean': te_mean,
            'Test Worst': te_worst, 'Test Gap': te_gap, 'Test Std': te_std
        })
        
    print("\n" + "=" * 80)
    print(f"SUMMARY (PyTorch Non-Linear Models — Test Set 2023-2025)")
    print("=" * 80)
    print(f"{'Method':<10} {'Train Mean':>12} {'Test Mean':>12} {'Worst Risk':>12} {'Risk Gap':>12} {'Std':>10}")
    print("-" * 80)
    
    with open(os.path.join(OUT_DIR, "irm_vrex_pytorch_results.txt"), 'w') as f:
        f.write("PyTorch Non-Linear IRM/V-REx Results\n")
        f.write("=" * 75 + "\n")
        f.write(f"Target: log(total_market_value)\n")
        f.write(f"{'Method':<10} {'Train Mean':>12} {'Test Mean':>12} {'Worst Risk':>12} {'Risk Gap':>12} {'Std':>10}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            line = f"{r['Method']:<10} {r['Train Mean']:>12.4f} {r['Test Mean']:>12.4f} {r['Test Worst']:>12.4f} {r['Test Gap']:>12.4f} {r['Test Std']:>10.4f}"
            print(line)
            f.write(line + "\n")

if __name__ == '__main__':
    main()
