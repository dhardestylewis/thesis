"""
evaluate_ood_generative_robust.py — Statistical Robustness Test for V-REx
=========================================================================
Tests the statistical significance of the V-REx OOD generalization by 
iterating over multiple random seeds and architectural variations.

Target: Amended Neighborhood Plan (OOD)
Metric: Downstream Logistic Regression PR-AUC (Mean & Std Dev)

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
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Configuration
SAMPLED_SIZE = 35000
EPOCHS = 80
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
MIN_ENV_SIZE = 5
VREX_PENALTY_WEIGHT = 10.0

SEEDS = [42, 101, 2024, 777, 1234]
ARCHITECTURES = [
    {"hidden1": 64, "hidden2": 32, "latent": 8},
    {"hidden1": 128, "hidden2": 64, "latent": 16},
    {"hidden1": 32, "hidden2": 16, "latent": 4}
]

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

# Ensure reproducibility within a single run
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CVAE(nn.Module):
    def __init__(self, input_dim, arch):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, arch['hidden1']),
            nn.SiLU(),
            nn.Linear(arch['hidden1'], arch['hidden2']),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(arch['hidden2'], arch['latent'])
        self.fc_logvar = nn.Linear(arch['hidden2'], arch['latent'])
        self.decoder = nn.Sequential(
            nn.Linear(arch['latent'] + 1, arch['hidden2']),
            nn.SiLU(),
            nn.Linear(arch['hidden2'], arch['hidden1']),
            nn.SiLU(),
            nn.Linear(arch['hidden1'], input_dim)
        )
    def encode(self, x, y):
        h = self.encoder(torch.cat([x, y.view(-1, 1)], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def decode(self, z, y):
        return self.decoder(torch.cat([z, y.view(-1, 1)], dim=1))
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

def elbo_loss(recon_x, x, mu, logvar):
    MSE = torch.sum((recon_x - x) ** 2, dim=1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return MSE + KLD

def train_cvae_vrex(model, train_loader, method="V-REx"):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    
    for epoch in range(EPOCHS):
        for x_batch, y_batch, env_batch in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch, y_batch)
            elbos = elbo_loss(recon_batch, x_batch, mu, logvar)
            
            unique_envs = torch.unique(env_batch)
            env_risks = [elbos[env_batch == e].mean() for e in unique_envs if (env_batch == e).sum() >= 2]
            
            if len(env_risks) < 2: continue
            
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

# --- Cached Data Loader ---
_CACHE = {}

def load_ood_data(seed):
    if seed in _CACHE: return _CACHE[seed]
    
    cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft', 
            'property_category_code', 'council_district', 'lui_general_land_use', 
            'year', 'protest', 'standardized_tcad_id']
            
    panel = pd.read_csv(PANEL_PATH, usecols=cols, low_memory=False)
    panel['improvement_sq_ft'] = pd.to_numeric(panel['improvement_sq_ft'], errors='coerce')
    panel = panel[panel['year'] <= 2024]
    
    env = pd.read_csv(ENV_PATH)
    env = env.rename(columns={'CASE_NUMBER': 'env_id'})
    
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BACKGROUND')
    df['SUB_TYPE'] = df['SUB_TYPE'].fillna('BACKGROUND')
    
    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]
    df['env_label'] = df['env_id'].map({name: i for i, name in enumerate(df['env_id'].unique())})
    
    # Introduce random seed to negative sample selection
    positives = df[df['protest'] == 1]
    negatives = df[df['protest'] == 0].sample(n=SAMPLED_SIZE - len(positives), random_state=seed)
    sampled_df = pd.concat([positives, negatives]).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    mask_test = sampled_df['SUB_TYPE'] == 'Amended Neighborhood Plan'
    train_df = sampled_df[~mask_test]
    test_df = sampled_df[mask_test]
    
    numeric_cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft']
    categorical_cols = ['property_category_code', 'council_district', 'lui_general_land_use']
    
    for col in numeric_cols: 
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df[col]  = test_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in categorical_cols: 
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)
        test_df[col]  = test_df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)
        
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    X_train = np.hstack([scaler.fit_transform(train_df[numeric_cols]), encoder.fit_transform(train_df[categorical_cols])])
    y_train = train_df['protest'].values.astype(np.float32)
    envs_train = train_df['env_label'].values.astype(np.int64)
    
    X_test = np.hstack([scaler.transform(test_df[numeric_cols]), encoder.transform(test_df[categorical_cols])])
    y_test = test_df['protest'].values.astype(np.float32)

    _CACHE[seed] = (X_train, y_train, envs_train, X_test, y_test)
    return _CACHE[seed]


def evaluate_ood_downstream(model, X_train, y_train, X_test, y_test, latent_dim):
    model.eval()
    
    with torch.no_grad():
        z_syn = torch.randn(5000, latent_dim)
        y_cond = torch.ones(5000)
        X_syn = model.decode(z_syn, y_cond).numpy()
        X_syn = np.nan_to_num(X_syn, posinf=0.0, neginf=0.0, nan=0.0)
    
    X_aug = np.vstack([X_train, X_syn])
    y_aug = np.concatenate([y_train, np.ones(5000)])
    
    clf_aug = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_aug.fit(X_aug, y_aug)
    try:
        return average_precision_score(y_test, clf_aug.predict_proba(X_test)[:, 1])
    except ValueError:
        return 0.0

def main():
    print(f"Running Statistical Robustness Test: {len(SEEDS)} Seeds, {len(ARCHITECTURES)} Architectures\n")
    
    results = [] # Stores (ERM_PR, VREX_PR) for every run
    
    for seed in SEEDS:
        X_train, y_train, envs_train, X_test, y_test = load_ood_data(seed)
        input_dim = X_train.shape[1]
        
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor(envs_train))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        for arch in ARCHITECTURES:
            set_seed(seed)
            model_erm = CVAE(input_dim, arch)
            train_cvae_vrex(model_erm, train_loader, method="ERM")
            pr_erm = evaluate_ood_downstream(model_erm, X_train, y_train, X_test, y_test, arch['latent'])
            
            set_seed(seed)
            model_vrex = CVAE(input_dim, arch)
            train_cvae_vrex(model_vrex, train_loader, method="V-REx")
            pr_vrex = evaluate_ood_downstream(model_vrex, X_train, y_train, X_test, y_test, arch['latent'])
            
            results.append((pr_erm, pr_vrex))
            print(f"Seed {seed:4d} | Arch: {arch['latent']:2d}-dim | ERM PR: {pr_erm:.4f} | V-REx PR: {pr_vrex:.4f} | Diff: {pr_vrex - pr_erm:+.4f}")
    
    # Statistical Summaries
    pr_erms = np.array([r[0] for r in results])
    pr_vrexs = np.array([r[1] for r in results])
    diffs = pr_vrexs - pr_erms
    
    print("\n" + "="*60)
    print("ROBUSTNESS TEST OOD RESULTS (N = 15 configurations)")
    print("="*60)
    print(f"ERM   Mean PR-AUC: {pr_erms.mean():.4f} +/- {pr_erms.std():.4f}")
    print(f"V-REx Mean PR-AUC: {pr_vrexs.mean():.4f} +/- {pr_vrexs.std():.4f}")
    print("-" * 60)
    print(f"Average Improvement (VREx - ERM): {diffs.mean():+.4f}")
    
    # Calculate Statistical Significance (1-sample t-test on diffs)
    t_stat = diffs.mean() / (diffs.std() / np.sqrt(len(diffs)))
    
    if diffs.mean() > 0 and t_stat > 1.76: # Approx alpha=0.05, 1-tailed for 14 df
        print(f"RESULT: STATISTICALLY SIGNIFICANT IMPROVEMENT (t={t_stat:.2f})")
    elif diffs.mean() > 0:
        print(f"RESULT: IMPROVEMENT NOT STATISTICALLY SIGNIFICANT (t={t_stat:.2f})")
    else:
        print(f"RESULT: V-REx FAILED TO GENERALIZE BETTER THAN ERM (t={t_stat:.2f})")

if __name__ == "__main__":
    main()
