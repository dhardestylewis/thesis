"""
generative_diffusion_vrex.py — Invariant Diffusion Representation Learning
==========================================================================
Integrates the V-REx invariance penalty into a Diffusion model. Instead of 
reconstruction ELBO (like CVAE), the model penalizes the variance of the 
denoising MSE (noise-prediction loss) across the 115 zoning environments.

Target: 'protest' (Binary Classification / Generation)
Metric: Downstream Logistic Regression on Oversampled Synthetics

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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import time
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
SAMPLED_SIZE = 33000
TIMESTEPS = 100
EPOCHS = 50
BATCH_SIZE = 512 # Smaller batch size because each diffusion step requires memory
LEARNING_RATE = 1e-3
MIN_ENV_SIZE = 5
VREX_PENALTY_WEIGHT = 1.0

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

# --- Diffusion MLP Architecture ---

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, time_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t.view(-1, 1).float())
        # Add time embedding to input
        t_emb = t_emb.expand(x.shape[0], -1) 
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)

# --- Diffusion Schedule (Linear) ---
beta = torch.linspace(1e-4, 0.02, TIMESTEPS)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bar[t].view(-1, 1)
    mean = torch.sqrt(alpha_bar_t) * x0
    var = torch.sqrt(1 - alpha_bar_t)
    return mean + var * noise, noise

# --- Training Routine ---

def train_diffusion_vrex(model, optimizer, train_loader, method="V-REx"):
    model.train()
    criterion = nn.MSELoss(reduction='none')
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_penalty = 0
        
        for batch in train_loader:
            x0 = batch[0]
            # y = batch[1] (Target class)
            envs = batch[2]
            
            # The input to diffusion is X conditionally augmented with Y
            # Actually, diffusion_benchmark.py joined X and y. Let's do that.
            y_col = batch[1].view(-1, 1)
            x0_joint = torch.cat([x0, y_col], dim=1)
            
            t = torch.randint(0, TIMESTEPS, (x0_joint.shape[0],)).long()
            
            # Add algorithmic noise
            xt, true_noise = forward_diffusion(x0_joint, t)
            
            # Predict noise
            noise_pred = model(xt, t)
            
            # Instance level predicting loss
            loss_i = criterion(noise_pred, true_noise).mean(dim=1)
            
            # Environment Level Risks
            unique_envs = torch.unique(envs)
            env_risks = []
            
            for e in unique_envs:
                mask = envs == e
                if mask.sum() >= 2:
                    env_risks.append(loss_i[mask].mean())
            
            if len(env_risks) < 2: continue
            
            env_risks_stack = torch.stack(env_risks)
            env_risks_stack = torch.clamp(env_risks_stack, max=50.0) # Prevent variance explosion
            erm_loss = env_risks_stack.mean()
            
            if method == "V-REx":
                penalty = env_risks_stack.var()
                # Anneal penalty: 0 for first 10 epochs, then linear increase
                if epoch < 10:
                    beta_weight = 0.0
                else:
                    beta_weight = VREX_PENALTY_WEIGHT * ((epoch - 10) / 40.0)
                final_loss = erm_loss + beta_weight * penalty
                total_penalty += penalty.item()
            else:
                final_loss = erm_loss
                
            optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Tighter clip
            optimizer.step()
            
            total_loss += final_loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Total Loss: {total_loss/len(train_loader):.4f} | Var Pen: {total_penalty/len(train_loader):.4f}")

# --- Data Loading ---

def load_data():
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
    
    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]
    
    env_map = {name: i for i, name in enumerate(df['env_id'].unique())}
    df['env_label'] = df['env_id'].map(env_map)
    
    positives = df[df['protest'] == 1]
    negatives = df[df['protest'] == 0].sample(n=SAMPLED_SIZE - len(positives), random_state=42)
    sampled_df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Sampled Dataset: {len(sampled_df):,} rows ({len(positives):,} positives)")
    
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


def evaluate_downstream(model, X_train, y_train, X_val, y_val, input_dim, method_name="Baseline"):
    model.eval()
    print(f"Generating 5,000 artificial protest samples ({method_name})...")
    
    with torch.no_grad():
        x = torch.randn(5000, input_dim) # Joint dimension (X + y)
        
        # Backward diffusion process
        for i in range(TIMESTEPS - 1, -1, -1):
            t = torch.tensor([i]).long().repeat(5000)
            noise_pred = model(x, t)
            
            alpha_t = alpha[i]
            alpha_bar_t = alpha_bar[i]
            beta_t = beta[i]
            
            if i > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
                
            mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))) * noise_pred)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z

    generated_data = x.numpy()
    generated_data = np.nan_to_num(generated_data, posinf=0.0, neginf=0.0, nan=0.0)
    
    # Isolate X and conditional y. 
    # Because we modeled the joint distribution [X, y], the model generates X and a scalar y.
    X_syn = generated_data[:, :-1]
    y_syn = generated_data[:, -1]
    
    # Filter only samples where the model generated a likely protest (y > 0.5)
    mask_pos = y_syn > 0.5
    X_syn_pos = X_syn[mask_pos]
    
    print(f"  Valid Protest Generates (y > 0.5): {len(X_syn_pos)}")
    
    # Real baseline
    clf_base = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_base.fit(X_train, y_train)
    probs_base = clf_base.predict_proba(X_val)[:, 1]
    auc_base = roc_auc_score(y_val, probs_base)
    pr_base = average_precision_score(y_val, probs_base)
    
    if len(X_syn_pos) < 10:
        return auc_base, pr_base, 0.0, 0.0
        
    # Augmented
    X_aug = np.vstack([X_train, X_syn_pos])
    y_aug = np.concatenate([y_train, np.ones(len(X_syn_pos))])
    
    clf_aug = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_aug.fit(X_aug, y_aug)
    probs_aug = clf_aug.predict_proba(X_val)[:, 1]
    auc_aug = roc_auc_score(y_val, probs_aug)
    pr_aug = average_precision_score(y_val, probs_aug)
    
    return auc_base, pr_base, auc_aug, pr_aug


def main():
    X_train, X_val, y_train, y_val, envs_train, envs_val = load_data()
    input_dim = X_train.shape[1] + 1 # +1 for joint y
    print(f"\nFeatures: {X_train.shape[1]} | Envs: {len(np.unique(envs_train))}")
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor(envs_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    results = []
    
    for method in ["ERM", "V-REx"]:
        print("\n" + "="*50)
        print(f"Training Diffusion with {method} Penalty")
        print("="*50)
        
        model = DiffusionModel(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        start_time = time.time()
        train_diffusion_vrex(model, optimizer, train_loader, method=method)
        print(f"Training finished in {time.time() - start_time:.2f}s")
        
        auc_b, pr_b, auc_a, pr_a = evaluate_downstream(
            model, X_train, y_train, X_val, y_val, input_dim, method_name=method
        )
        
        results.append({
            'Method': method,
            'Real_ROC': auc_b, 'Real_PR': pr_b,
            'Aug_ROC': auc_a,  'Aug_PR': pr_a
        })

    print("\n" + "="*80)
    print("FINAL DOWNSTREAM LOGISTIC REGRESSION PERFORMANCE (Diffusion Syn. Augmentation)")
    print("="*80)
    print(f"{'Method/Architecture':<25} {'Real ROC':>12} {'Aug ROC':>12} | {'Real PR':>12} {'Aug PR':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Method']+' Diffusion':<25} {r['Real_ROC']:>12.4f} {r['Aug_ROC']:>12.4f} | {r['Real_PR']:>12.4f} {r['Aug_PR']:>12.4f}")


if __name__ == "__main__":
    main()
