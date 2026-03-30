"""
generative_cvae_vrex.py — Invariant Generative Representation Learning
======================================================================
Tests if a Conditional Variational Autoencoder (CVAE) can learn an invariant
latent representation (Z) across 115 Austin zoning environments. By penalizing
the variance of the Evidence Lower Bound (ELBO) across environments (V-REx
penalty), the model discards spurious correlations that only explain 'protest'
within localized multi-parcel interventions.

Target: 'protest' (Binary Classification / Generation)
Metric: Downstream Logistic Regression on Oversampled Z (Synthetic Data)

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import os
import time
import warnings

warnings.filterwarnings('ignore')

# Configuration
SAMPLED_SIZE = 33000
LATENT_DIM = 8
EPOCHS = 100
BATCH_SIZE = 1024  # Increased for environment risk variance estimation
LEARNING_RATE = 1e-3
MIN_ENV_SIZE = 5
VREX_PENALTY_WEIGHT = 10.0

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_v3.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

# --- Model Definition ---

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CVAE, self).__init__()
        # Encoder (Conditioned on Y)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder (Conditioned on Y)
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
    """Returns vector of ELBOs per instance."""
    # Mean Squared Error per sample across features
    MSE = torch.sum((recon_x - x) ** 2, dim=1)
    
    # KL Divergence per sample
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    return MSE + KLD

# --- Training Routine ---

def train_cvae_vrex(model, optimizer, train_loader, method="V-REx"):
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_penalty = 0
        
        for x_batch, y_batch, env_batch in train_loader:
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(x_batch, y_batch)
            
            # Instance-level ELBOs
            elbos = elbo_loss(recon_batch, x_batch, mu, logvar)
            
            # Map risks to environments
            unique_envs = torch.unique(env_batch)
            env_risks = []
            
            for e in unique_envs:
                mask = env_batch == e
                if mask.sum() >= 2:
                    env_risks.append(elbos[mask].mean())
            
            if not env_risks:
                continue
                
            env_risks_stack = torch.stack(env_risks)
            erm_loss = env_risks_stack.mean()
            
            if method == "V-REx":
                penalty = env_risks_stack.var()
                # Anneal penalty: full strength after epoch 20
                beta = VREX_PENALTY_WEIGHT if epoch > 20 else VREX_PENALTY_WEIGHT * (epoch / 20.0)
                final_loss = erm_loss + beta * penalty
                total_penalty += penalty.item()
            else:
                final_loss = erm_loss
                
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            total_loss += final_loss.item()
            
            # Logging metrics
            mse = torch.sum((recon_batch - x_batch) ** 2, dim=1).mean().item()
            kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean().item()
            total_recon += mse
            total_kl += kl
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Total Loss: {total_loss/len(train_loader):.4f} "
                  f"(MSE: {total_recon/len(train_loader):.4f}, KL: {total_kl/len(train_loader):.4f}) "
                  f"| Var Pen: {total_penalty/len(train_loader):.4f}")

# --- Data Loading ---

def load_data():
    print("Loading data...")
    cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft', 
            'property_category_code', 'council_district', 'lui_general_land_use', 
            'year', 'protest', 'standardized_tcad_id']
            
    # Remove strict type for improvement_sq_ft due to strings like 'C-5'
    panel = pd.read_csv(PANEL_PATH, usecols=cols, low_memory=False)
    panel['improvement_sq_ft'] = pd.to_numeric(panel['improvement_sq_ft'], errors='coerce')
    panel = panel[panel['year'] <= 2024]
    
    print("Loading environment assignments...")
    env = pd.read_csv(ENV_PATH)
    env = env.rename(columns={'CASE_NUMBER': 'env_id'})
    
    # Merge and identify environments (keeping untreated as part of 'background' env)
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BACKGROUND')
    
    # Filter valid environments
    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]
    
    env_map = {name: i for i, name in enumerate(df['env_id'].unique())}
    df['env_label'] = df['env_id'].map(env_map)
    
    # Sampling Strategy: All positives + subsampled negatives
    positives = df[df['protest'] == 1]
    negatives = df[df['protest'] == 0].sample(n=SAMPLED_SIZE - len(positives), random_state=42)
    sampled_df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Sampled Dataset: {len(sampled_df):,} rows ({len(positives):,} positives)")
    
    # Featurization
    numeric_cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft']
    categorical_cols = ['property_category_code', 'council_district', 'lui_general_land_use']
    
    for col in numeric_cols: sampled_df[col] = sampled_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in categorical_cols: sampled_df[col] = sampled_df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(sampled_df[numeric_cols])
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(sampled_df[categorical_cols])
    
    X = np.hstack([X_num, X_cat])
    y = sampled_df['protest'].values.astype(np.float32)
    envs = sampled_df['env_label'].values.astype(np.int64)
    
    return train_test_split(X, y, envs, test_size=0.2, random_state=42, stratify=y)


def evaluate_downstream(model, X_train, y_train, X_val, y_val, method_name="Baseline"):
    model.eval()
    print(f"\nEvaluating {method_name} Synthetic Amplification...")
    
    # Generate 5,000 synthetic protest=1 samples
    with torch.no_grad():
        z_syn = torch.randn(5000, LATENT_DIM)
        y_cond = torch.ones(5000)
        X_syn = model.decode(z_syn, y_cond).numpy()
        
    # Sanitize synthetic generations (CVAE decoder can occasionally output inf or extreme floats)
    X_syn = np.nan_to_num(X_syn, posinf=0.0, neginf=0.0, nan=0.0)
    
    # 1. Real Only Baseline
    clf_base = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_base.fit(X_train, y_train)
    probs_base = clf_base.predict_proba(X_val)[:, 1]
    auc_base = roc_auc_score(y_val, probs_base)
    pr_base = average_precision_score(y_val, probs_base)
    
    # 2. Augmented (Real + Synthetic)
    X_aug = np.vstack([X_train, X_syn])
    y_aug = np.concatenate([y_train, np.ones(5000)])
    
    clf_aug = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_aug.fit(X_aug, y_aug)
    probs_aug = clf_aug.predict_proba(X_val)[:, 1]
    auc_aug = roc_auc_score(y_val, probs_aug)
    pr_aug = average_precision_score(y_val, probs_aug)
    
    print(f"  Real Only ROC-AUC: {auc_base:.4f}  |  PR-AUC: {pr_base:.4f}")
    print(f"  Augmented ROC-AUC: {auc_aug:.4f}  |  PR-AUC: {pr_aug:.4f}")
    
    return auc_base, pr_base, auc_aug, pr_aug


def main():
    X_train, X_val, y_train, y_val, envs_train, envs_val = load_data()
    input_dim = X_train.shape[1]
    print(f"\nFeatures: {input_dim} | Envs: {len(np.unique(envs_train))}")
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor(envs_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    results = []
    
    for method in ["ERM", "V-REx"]:
        print("\n" + "="*50)
        print(f"Training CVAE with {method} Penalty")
        print("="*50)
        
        model = CVAE(input_dim, LATENT_DIM)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        start_time = time.time()
        train_cvae_vrex(model, optimizer, train_loader, method=method)
        print(f"Training finished in {time.time() - start_time:.2f}s")
        
        auc_b, pr_b, auc_a, pr_a = evaluate_downstream(
            model, X_train, y_train, X_val, y_val, method_name=method
        )
        
        results.append({
            'Method': method,
            'Real_ROC': auc_b, 'Real_PR': pr_b,
            'Aug_ROC': auc_a,  'Aug_PR': pr_a
        })
        
    print("\n" + "="*80)
    print("FINAL DOWNSTREAM LOGISTIC REGRESSION PERFORMANCE (Synthetic Data Augmentation)")
    print("="*80)
    print(f"{'Method/Architecture':<25} {'Real ROC':>12} {'Aug ROC':>12} | {'Real PR':>12} {'Aug PR':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Method']+' Generative':<25} {r['Real_ROC']:>12.4f} {r['Aug_ROC']:>12.4f} | {r['Real_PR']:>12.4f} {r['Aug_PR']:>12.4f}")

if __name__ == "__main__":
    main()
