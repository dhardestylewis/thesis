"""
evaluate_ood_modal_fast.py
==========================
Parallel computing script built on Modal to train 30 generative models
virtually instantly across 30 T4 GPUs.

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""

import modal
import os
import itertools
import numpy as np

TENSORS_DIR = r"c:\Users\dhl\data\thesis\thesis\Analysis\Data\Tensors"

app = modal.App("thesis-vrex-robustness")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "scikit-learn", "numpy", "pandas")
    .add_local_dir(TENSORS_DIR, remote_path="/data")
)

@app.function(image=image, gpu="T4", timeout=300)
def evaluate_single_run(config):
    # Unpack config
    seed = config["seed"]
    arch_id = config["arch_id"]
    method = config["method"]
    arch = config["arch"]
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    import sys
    
    # Configuration
    EPOCHS = 80
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-3
    VREX_PENALTY_WEIGHT = 10.0

    # Ensure reproducibility within a single run
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    # Load tensors from the mount
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.load("/data/X_train.pt").to(device)
    y_train = torch.load("/data/y_train.pt").to(device)
    envs_train = torch.load("/data/envs_train.pt").to(device)
    
    X_test_pt = torch.load("/data/X_test.pt")
    X_test = X_test_pt.numpy()
    y_test = torch.load("/data/y_test.pt").numpy()
    
    input_dim = X_train.shape[1]
    
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

    # Initialize Model on GPU
    model = CVAE(input_dim, arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create DataLoader
    train_ds = TensorDataset(X_train, y_train, envs_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training Loop
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
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        z_syn = torch.randn(5000, arch['latent']).to(device)
        y_cond = torch.ones(5000).to(device)
        X_syn = model.decode(z_syn, y_cond).cpu().numpy()
        X_syn = np.nan_to_num(X_syn, posinf=0.0, neginf=0.0, nan=0.0)
        
    X_train_cpu = X_train.cpu().numpy()
    y_train_cpu = y_train.cpu().numpy()
    X_aug = np.vstack([X_train_cpu, X_syn])
    y_aug = np.concatenate([y_train_cpu, np.ones(5000)])
    
    clf_aug = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_aug.fit(X_aug, y_aug)
    
    try:
        pr_auc = average_precision_score(y_test, clf_aug.predict_proba(X_test)[:, 1])
    except ValueError:
        pr_auc = 0.0
        
    return {
        "seed": seed,
        "arch_id": arch_id,
        "latent_dim": arch["latent"],
        "method": method,
        "pr_auc": float(pr_auc)
    }


@app.local_entrypoint()
def run_all():
    SEEDS = [42, 101, 2024, 777, 1234]
    METHODS = ["ERM", "V-REx"]
    ARCHITECTURES = [
        {"hidden1": 64, "hidden2": 32, "latent": 8},
        {"hidden1": 128, "hidden2": 64, "latent": 16},
        {"hidden1": 32, "hidden2": 16, "latent": 4}
    ]
    
    # Generate the 30 configurations
    configs = []
    for seed in SEEDS:
        for arch_id, arch in enumerate(ARCHITECTURES):
            for method in METHODS:
                configs.append({
                    "seed": seed,
                    "arch_id": arch_id,
                    "method": method,
                    "arch": arch
                })
                
    print(f"Launching {len(configs)} Generative Models in parallel on Modal GPUs...")
    
    # Run in parallel
    results = list(evaluate_single_run.map(configs))
    
    # Process results into a DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(r"c:\Users\dhl\data\thesis\thesis\Analysis\Results\modal_robustness_results.csv", index=False)
    
    # Re-align pairs
    paired = df.pivot(index=["seed", "arch_id", "latent_dim"], columns="method", values="pr_auc").reset_index()
    paired["diff"] = paired["V-REx"] - paired["ERM"]
    
    print("\n" + "="*80)
    print("MODAL ROBUSTNESS OOD RESULTS (N = 15 Architecture/Seed Variations)")
    print("="*80)
    for _, row in paired.iterrows():
        print(f"Seed {int(row['seed']):4d} | Arch: {int(row['latent_dim']):2d}-dim | ERM PR: {row['ERM']:.4f} | V-REx PR: {row['V-REx']:.4f} | Diff: {row['diff']:+.4f}")
        
    print("-" * 80)
    erm_mean = paired["ERM"].mean()
    erm_std = paired["ERM"].std()
    vrex_mean = paired["V-REx"].mean()
    vrex_std = paired["V-REx"].std()
    diff_mean = paired["diff"].mean()
    diff_std = paired["diff"].std()
    
    print(f"ERM   Mean PR-AUC: {erm_mean:.4f} +/- {erm_std:.4f}")
    print(f"V-REx Mean PR-AUC: {vrex_mean:.4f} +/- {vrex_std:.4f}")
    print(f"Average Improvement (VREx - ERM): {diff_mean:+.4f}")
    
    import numpy as np
    t_stat = diff_mean / (diff_std / np.sqrt(len(paired)))
    
    if diff_mean > 0 and t_stat > 1.76: # Approx alpha=0.05, 1-tailed for 14 df
        print(f"\nRESULT: STATISTICALLY SIGNIFICANT IMPROVEMENT (t={t_stat:.2f})")
    elif diff_mean > 0:
        print(f"\nRESULT: IMPROVEMENT NOT STATISTICALLY SIGNIFICANT (t={t_stat:.2f})")
    else:
        print(f"\nRESULT: V-REx FAILED TO GENERALIZE BETTER THAN ERM (t={t_stat:.2f})")
