import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

import os
import time

# Configuration
sampled_size = 33000
latent_dim = 8
epochs = 30
batch_size = 64
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
df = pd.read_csv("Data/Panel/Output/Property_Year_Panel_Enriched.csv", usecols=cols, dtype=types)
df['improvement_sq_ft'] = pd.to_numeric(df['improvement_sq_ft'], errors='coerce').astype('float32')

# Filter for relevant years (2019-2024 for training/val)
df = df[df['year'] <= 2024]

# Sampling Strategy: All positives + subsampled negatives
positives = df[df['protest'] == 1]
negatives = df[df['protest'] == 0].sample(n=sampled_size - len(positives), random_state=42)
sampled_df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Sampled Dataset: {len(sampled_df)} rows ({len(positives)} positives)")

# Feature Engineering (Simplified)
numeric_cols = ['total_market_value', 'deed_acreage', 'improvement_sq_ft'] # Add more if available/relevant and standardized
categorical_cols = ['property_category_code', 'council_district', 'lui_general_land_use']

# Handle missing values (simple fill for prototype)
for col in numeric_cols:
    sampled_df[col] = sampled_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
for col in categorical_cols:
    sampled_df[col] = sampled_df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing')

# Preprocessing
scaler = StandardScaler()
X_num = scaler.fit_transform(sampled_df[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(sampled_df[categorical_cols])

X = np.hstack([X_num, X_cat])
y = sampled_df['protest'].values

input_dim = X.shape[1]
print(f"Input Dimension: {input_dim}")

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch Tensors
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# CVAE Model Definition
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 32), # +1 for condition (y)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 16), # +1 for condition (y)
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim) # Reconstruct X
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

# Condition Prediction Network (Probability of y given x)
# Wait, standard CVAE models P(X|Y). To predict Y|X, we can use Bayes: P(Y|X) = P(X|Y)P(Y) / P(X)
# OR we can train a separate classifier on the latent space.
# For this prototype, let's use the latent space to train a simple Logistic Regression outcomes
# This demonstrates the "invariant representation" idea.

# Loss Function
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Training Loop
model = CVAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training CVAE...")
start_time = time.time()
train_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, target)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

print(f"Training finished in {time.time() - start_time:.2f}s")

# Evaluation: Latent Space Utility
# We extract latent representations z for all data (using y=cond)
# Then train a classifier on z to predict y.
# This tests if the latent space captures relevant info for prediction.

print("extracting latent features...")
model.eval()
with torch.no_grad():
    # For training data
    mu_train, _ = model.encode(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    Z_train = mu_train.numpy()
    
    # For validation data - wait, at test time we don't know Y.
    # To predict Y, we can't condition on it in the encoder.
    # Approach 2: Use the encoder to map X -> Z (unconditional encoder?) or train a separate P(Y|X) network?
    # Actually, for CVAE-based anomaly detection/classification:
    # We can evaluate ELBO for y=0 and y=1 and pick the higher one.
    # P(y=1|x) approx P(x|y=1)P(y=1) / sum_c P(x|y=c)P(y=c)
    
    # Let's try the ELBO method for classification
    y_scores = []
    
    # Prior P(y=1)
    p_y1 = np.mean(y_train)
    p_y0 = 1 - p_y1
    
    for i in range(len(X_val)):
        x_sample = torch.FloatTensor(X_val[i:i+1])
        
        # Cond on y=0
        y0 = torch.zeros(1)
        recon_0, mu_0, logvar_0 = model(x_sample, y0)
        loss_0 = loss_function(recon_0, x_sample, mu_0, logvar_0).item()
        
        # Cond on y=1
        y1 = torch.ones(1)
        recon_1, mu_1, logvar_1 = model(x_sample, y1)
        loss_1 = loss_function(recon_1, x_sample, mu_1, logvar_1).item()
        
        # Convert negative ELBO (loss) to pseudo-probability scores
        # Lower loss = higher likelihood
        # Score = likelihood_1 / (likelihood_1 + likelihood_0)
        # working in log space: log_p0 = -loss_0, log_p1 = -loss_1
        # P(y=1|x) = exp(-loss_1) * p_y1 / (exp(-loss_1)*p_y1 + exp(-loss_0)*p_y0)
        
        # Simplify: score is related to difference in reconstruction error
        # If x fits better with y=1 condition, it's likely y=1.
        
        # Let's compute robust logistic probability
        # log_prob_1 = -loss_1 + np.log(p_y1)
        # log_prob_0 = -loss_0 + np.log(p_y0)
        # prob_1 = 1 / (1 + exp(log_prob_0 - log_prob_1))
        
        logit = (-loss_1 + np.log(p_y1)) - (-loss_0 + np.log(p_y0))
        prob = 1 / (1 + np.exp(-logit))
        y_scores.append(prob)

    y_scores = np.array(y_scores)
    
    auc = roc_auc_score(y_val, y_scores)
    pr_auc = average_precision_score(y_val, y_scores)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print(f"Validation PR-AUC: {pr_auc:.4f}")

    # Calibration Check
    print("Calibration (Brier Score):", brier_score_loss(y_val, y_scores))
    
    # Save results
    results_df = pd.DataFrame({'y_true': y_val, 'y_prob': y_scores})
    results_df.to_csv("Analysis/Results/Benchmarks/cvae_benchmark_results.csv", index=False)
    
    print("Results saved.")

