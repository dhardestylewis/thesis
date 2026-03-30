"""
evaluate_ood_generative.py — Out-of-Distribution Invariance Testing
===================================================================
Tests whether the V-REx representation penalty allows a generative model 
to generalize to Out-of-Distribution (OOD) zoning interventions.

1. Train: Non-PUD Environments (e.g. Amended Neighborhood Plans, MUDs)
2. Test: Extrapolation strictly to withheld 'Planned Unit Development' (PUDs)
3. Metric: Downstream Logistic Regression ROC-AUC on PUDs.

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import time
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
SAMPLED_SIZE = 35000
LATENT_DIM = 8
EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
MIN_ENV_SIZE = 5
VREX_PENALTY_WEIGHT = 10.0

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_v3.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

# --- Model Definition (CVAE for faster iteration) ---

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, input_dim)
        )
        
    def encode(self, x, y):
        inputs = torch.cat([x, y.view(-1, 1)], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        inputs = torch.cat([z, y.view(-1, 1)], dim=1)
        return self.decoder(inputs)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

def elbo_loss(recon_x, x, mu, logvar):
    MSE = torch.sum((recon_x - x) ** 2, dim=1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return MSE + KLD

# --- Training Routine ---

def train_cvae_vrex(model, optimizer, train_loader, method="V-REx"):
    model.train()
    
    for epoch in range(EPOCHS):
        for x_batch, y_batch, env_batch in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch, y_batch)
            elbos = elbo_loss(recon_batch, x_batch, mu, logvar)
            
            unique_envs = torch.unique(env_batch)
            env_risks = []
            
            for e in unique_envs:
                mask = env_batch == e
                if mask.sum() >= 2:
                    env_risks.append(elbos[mask].mean())
            
            if not env_risks: continue
                
            env_risks_stack = torch.stack(env_risks)
            erm_loss = env_risks_stack.mean()
            
            if method == "V-REx":
                penalty = env_risks_stack.var()
                beta = VREX_PENALTY_WEIGHT if epoch > 20 else VREX_PENALTY_WEIGHT * (epoch / 20.0)
                final_loss = erm_loss + beta * penalty
            else:
                final_loss = erm_loss
                
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

# --- Data Loading (OOD Splitting) ---

def load_ood_data():
    print("Loading data...")
    cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft', 
            'property_category_code', 'council_district', 'lui_general_land_use', 
            'year', 'protest', 'standardized_tcad_id']
            
    panel = pd.read_csv(PANEL_PATH, usecols=cols, low_memory=False)
    panel['improvement_sq_ft'] = pd.to_numeric(panel['improvement_sq_ft'], errors='coerce')
    panel = panel[panel['year'] <= 2024]
    
    print("Loading environment assignments...")
    env = pd.read_csv(ENV_PATH)
    env = env.rename(columns={'CASE_NUMBER': 'env_id'})
    
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BACKGROUND')
    df['SUB_TYPE'] = df['SUB_TYPE'].fillna('BACKGROUND')
    
    # Filter valid environments by size
    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]
    
    env_map = {name: i for i, name in enumerate(df['env_id'].unique())}
    df['env_label'] = df['env_id'].map(env_map)
    
    # Sampling Strategy to balance compute
    positives = df[df['protest'] == 1]
    negatives = df[df['protest'] == 0].sample(n=SAMPLED_SIZE - len(positives), random_state=42)
    sampled_df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # === OUT OF DISTRIBUTION CONCEPT SPLIT ===
    # Train/Val: Everything EXCEPT "Amended Neighborhood Plan"
    # Test: STRICTLY "Amended Neighborhood Plan"
    
    mask_test = sampled_df['SUB_TYPE'] == 'Amended Neighborhood Plan'
    train_df = sampled_df[~mask_test]
    test_df = sampled_df[mask_test]
    
    print(f"\nOOD Split:")
    print(f"  Train/Val (Non-Neighborhood Plans + Background): {len(train_df):,} rows")
    print(f"  OOD Test (Strictly Neighborhood Plans): {len(test_df):,} rows")
    print(f"    (Test Positives: {test_df['protest'].sum():.0f})\n")
    
    if test_df['protest'].sum() < 5:
        print("WARNING: Very few protest incidents inside the OOD PUD test set.")

    # Featurization
    numeric_cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft']
    categorical_cols = ['property_category_code', 'council_district', 'lui_general_land_use']
    
    # Fit scalers only on training data
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    for col in numeric_cols: 
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df[col]  = test_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
    for col in categorical_cols: 
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)
        test_df[col]  = test_df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)
    
    # Transform Training Data
    X_train_num = scaler.fit_transform(train_df[numeric_cols])
    X_train_cat = encoder.fit_transform(train_df[categorical_cols])
    X_train = np.hstack([X_train_num, X_train_cat])
    y_train = train_df['protest'].values.astype(np.float32)
    envs_train = train_df['env_label'].values.astype(np.int64)
    
    # Transform OOD Test Data
    X_test_num = scaler.transform(test_df[numeric_cols])
    X_test_cat = encoder.transform(test_df[categorical_cols])
    X_test = np.hstack([X_test_num, X_test_cat])
    y_test = test_df['protest'].values.astype(np.float32)
    
    return X_train, y_train, envs_train, X_test, y_test


def evaluate_ood_downstream(model, X_train, y_train, X_test, y_test, method_name="Baseline"):
    model.eval()
    
    # 1. Real Only Baseline (ERM purely on non-PUD features)
    clf_base = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_base.fit(X_train, y_train)
    probs_base = clf_base.predict_proba(X_test)[:, 1]
    
    try:
        auc_base = roc_auc_score(y_test, probs_base)
        pr_base = average_precision_score(y_test, probs_base)
    except ValueError:
        # Fails if the test set entirely lacks one class 
        auc_base = 0.0
        pr_base = 0.0
    
    # 2. Generative Augmentation 
    with torch.no_grad():
        z_syn = torch.randn(5000, LATENT_DIM)
        y_cond = torch.ones(5000)
        X_syn = model.decode(z_syn, y_cond).numpy()
        X_syn = np.nan_to_num(X_syn, posinf=0.0, neginf=0.0, nan=0.0)
    
    X_aug = np.vstack([X_train, X_syn])
    y_aug = np.concatenate([y_train, np.ones(5000)])
    
    clf_aug = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_aug.fit(X_aug, y_aug)
    probs_aug = clf_aug.predict_proba(X_test)[:, 1]
    
    try:
        auc_aug = roc_auc_score(y_test, probs_aug)
        pr_aug = average_precision_score(y_test, probs_aug)
    except ValueError:
        auc_aug = 0.0
        pr_aug = 0.0
    
    return auc_base, pr_base, auc_aug, pr_aug


def main():
    X_train, y_train, envs_train, X_test, y_test = load_ood_data()
    input_dim = X_train.shape[1]
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor(envs_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    results = []
    
    for method in ["ERM", "V-REx"]:
        print(f"Training Generative {method}...")
        model = CVAE(input_dim, LATENT_DIM)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        start_time = time.time()
        train_cvae_vrex(model, optimizer, train_loader, method=method)
        
        auc_b, pr_b, auc_a, pr_a = evaluate_ood_downstream(
            model, X_train, y_train, X_test, y_test, method_name=method
        )
        
        results.append({
            'Method': method,
            'Real_ROC': auc_b, 'Real_PR': pr_b,
            'Aug_ROC': auc_a,  'Aug_PR': pr_a
        })
        
    print("\n" + "="*80)
    print("OOD EXTRAPOLATION TEST: Predict Protests strictly on Withheld PUDs")
    print("="*80)
    print(f"{'Method/Architecture':<25} {'Real ROC':>12} {'Aug ROC':>12} | {'Real PR':>12} {'Aug PR':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Method']+' Augmentation':<25} {r['Real_ROC']:>12.4f} {r['Aug_ROC']:>12.4f} | {r['Real_PR']:>12.4f} {r['Aug_PR']:>12.4f}")

if __name__ == "__main__":
    main()
