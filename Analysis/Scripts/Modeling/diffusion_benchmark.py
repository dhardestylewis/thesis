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

# Configuration
sampled_size = 33000
batch_size = 64
timesteps = 100 # Reduced for prototype (usually 1000)
epochs = 50
learning_rate = 1e-3

# Optimized Load
cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft', 
        'property_category_code', 'council_district', 'lui_general_land_use', 
        'year', 'protest']

types = {
    'total_market_value': 'float32',
    'deed_acreage': 'float32',
    # 'improvement_sq_ft': 'float32', # Remove strict type due to dirty data (e.g. 'C-5')
    'property_category_code': 'str',
    'council_district': 'str',
    'lui_general_land_use': 'str',
    'year': 'int32',
    'protest': 'int8'
}

print("Loading data (optimized)...")
df = pd.read_csv("Data/Panel/Output/Property_Year_Panel.csv", usecols=cols, dtype=types)
df['improvement_sq_ft'] = pd.to_numeric(df['improvement_sq_ft'], errors='coerce').astype('float32')
df = df[df['year'] <= 2024]

# Sampling
positives = df[df['protest'] == 1]
negatives = df[df['protest'] == 0].sample(n=sampled_size - len(positives), random_state=42)
sampled_df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Sampled Dataset: {len(sampled_df)} rows")

# Preprocessing
numeric_cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft']
categorical_cols = ['property_category_code', 'council_district', 'lui_general_land_use']

for col in numeric_cols: sampled_df[col] = sampled_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
for col in categorical_cols: sampled_df[col] = sampled_df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing')

scaler = StandardScaler()
X_num = scaler.fit_transform(sampled_df[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(sampled_df[categorical_cols])

X = np.hstack([X_num, X_cat])
y = sampled_df['protest'].values
input_dim = X.shape[1] + 1 # +1 for y (joined)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Join X and y for joint distribution modeling
train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_data)), batch_size=batch_size, shuffle=True)

# MLP Diffusion Model (Simple Backbone)
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
        t_emb = self.time_embed(t)
        # Add time embedding to input (broadcast)
        # Simple concat approach
        t_emb = t_emb.expand(x.shape[0], -1) 
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)

# Diffusion Schedule (Linear)
beta = torch.linspace(1e-4, 0.02, timesteps)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bar[t].view(-1, 1)
    mean = torch.sqrt(alpha_bar_t) * x0
    var = torch.sqrt(1 - alpha_bar_t)
    return mean + var * noise, noise

model = DiffusionModel(input_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training Diffusion Model...")
start_time = time.time()
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        x0 = batch[0]
        t = torch.randint(0, timesteps, (x0.shape[0], 1)).float()
        
        # Add noise
        xt, noise = forward_diffusion(x0, t.long().flatten())
        
        # Predict noise
        noise_pred = model(xt, t)
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

print(f"Training finished in {time.time() - start_time:.2f}s")

# Sampling / Oversampling
print("Generating 2,000 synthetic protest samples...")
model.eval()
with torch.no_grad():
    x = torch.randn(2000, input_dim) # Start from noise
    
    for i in range(timesteps - 1, -1, -1):
        t = torch.tensor([i]).float().repeat(2000, 1)
        
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

    # Generate samples are continuous. We need to threshold y column.
    generated_data = x.numpy()
    
    # Filter for those with y > 0.5 (synthetic positives)
    # Actually, we modeled joint distribution. The model might produce y values anywhere.
    # But since we want "conditional generation", a better approach is GUIDED sampling, but standard sampling works too
    # Let's assess the quality of generated 'y' channel.
    
    print("Generated data shape:", generated_data.shape)
    
    # Separate X and y
    X_syn = generated_data[:, :-1]
    y_syn = generated_data[:, -1]
    
    # Train a downstream classifier
    print("Training downstream classifier on Real + Synthetic...")
    
    # 1. Real Only Baseline
    clf_base = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_base.fit(X_train, y_train)
    probs_base = clf_base.predict_proba(X_val)[:, 1]
    print(f"Baseline (Real) ROC-AUC: {roc_auc_score(y_val, probs_base):.4f}")
    
    # 2. Augmented
    # Treat generated samples with y > 0.5 as positives
    mask_pos = y_syn > 0.5
    X_aug = np.vstack([X_train, X_syn[mask_pos]])
    y_aug = np.concatenate([y_train, np.ones(mask_pos.sum())])
    
    clf_aug = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_aug.fit(X_aug, y_aug)
    probs_aug = clf_aug.predict_proba(X_val)[:, 1]
    
    print(f"Augmented (Diff) ROC-AUC: {roc_auc_score(y_val, probs_aug):.4f}")
    print(f"Augmented (Diff) PR-AUC: {average_precision_score(y_val, probs_aug):.4f}")
    
    # Save results
    results_df = pd.DataFrame({'y_true': y_val, 'y_prob_base': probs_base, 'y_prob_aug': probs_aug})
    results_df.to_csv("Analysis/Results/Benchmarks/diffusion_benchmark_results.csv", index=False)
