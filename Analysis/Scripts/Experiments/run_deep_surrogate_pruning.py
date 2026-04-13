import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

print("\n" + "="*70)
print(" PYTORCH DEEP SURROGATE: LATENT NEURON PRUNING AUDIT")
print("="*70)

DATA_FILE = r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv'

# Based on VR-Ex architecture from shap_vrex_final.py
LATENT = 8

class DeepSurrogateProxy(nn.Module):
    """
    Simplified proxy of the VR-Ex Deep Surrogate with a hook for latent masking.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU()
        )
        self.latent_proj = nn.Linear(128, LATENT)
        self.clf = nn.Linear(LATENT, 1)

    def forward(self, x, latent_mask=None):
        features = self.enc(x)
        z = self.latent_proj(features)
        
        # Apply structured pruning mask to the latent representation
        if latent_mask is not None:
            z = z * latent_mask
            
        logits = self.clf(z)
        return torch.sigmoid(logits)

def main():
    print("[1] Initializing PyTorch Pruning Infrastructure...")
    df = pd.read_csv(DATA_FILE, low_memory=False).dropna(subset=['year', 'is_protested'])
    df['is_protested'] = pd.to_numeric(df['is_protested'], errors='coerce')
    
    # We strip text and ids. We assume standardized numerical inputs for the deep model.
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'target']
    leak_cols = [c for c in df.columns if c.startswith('tfidf_') or c.startswith('speech_') or c in drop_cols]
    
    X = df.drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
    X = X.loc[:, X.nunique() > 1].fillna(0)
    y = df['is_protested'].astype(int).values
    
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X.values)
    
    # Simple split
    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    x_tr_T = torch.FloatTensor(X_tr)
    y_tr_T = torch.FloatTensor(y_tr).unsqueeze(1)
    x_val_T = torch.FloatTensor(X_val)

    model = DeepSurrogateProxy(input_dim=x_tr_T.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    print(f"[2] Training Full Baseline Network (Features: {x_tr_T.shape[1]}, Latent Dim: {LATENT})...")
    model.train()
    for epoch in range(150):
        opt.zero_grad()
        out = model(x_tr_T)
        loss = loss_fn(out, y_tr_T)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        base_preds = model(x_val_T).squeeze().numpy()
        baseline_prauc = average_precision_score(y_val, base_preds)
        print(f"    Baseline Validation PR-AUC: {baseline_prauc:.4f}")

    print("\n[3] Executing Iterative Latent Neuron Knockout:")
    print("    Systematically masking each of the 32 bottleneck neurons to map representation harm.")
    print(f"{'Ablated Latent Neuron':<35} | {'Validation PR-AUC':>20} | {'Delta':>10}")
    print("-" * 73)
    
    # Create the baseline unmasked tensor
    baseline_mask = torch.ones(LATENT)

    results = []
    with torch.no_grad():
        for i in range(LATENT):
            # Create a mask that zeros out exactly ONE latent neuron
            mask = baseline_mask.clone()
            mask[i] = 0.0
            
            # Forward pass through the validation set with the broken circuit
            ko_preds = model(x_val_T, latent_mask=mask).squeeze().numpy()
            ko_prauc = average_precision_score(y_val, ko_preds)
            
            delta = ko_prauc - baseline_prauc
            results.append((i, delta))
            
            if delta > 0.001:
                 print(f"Latent Node [{i:02d}] (Masked)                | {ko_prauc:>20.4f} | +{delta:>9.4f} [HARMFUL]")
            elif delta < -0.01:
                 print(f"Latent Node [{i:02d}] (Masked)                | {ko_prauc:>20.4f} | {delta:>10.4f} [CRITICAL]")
            else:
                 print(f"Latent Node [{i:02d}] (Masked)                | {ko_prauc:>20.4f} | {delta:>10.4f}")

    print("-" * 73)
    harmful_nodes = [r for r in results if r[1] > 0]
    critical_nodes = sorted([r for r in results if r[1] < 0], key=lambda x: x[1])[:5]
    
    print("\n[CONCLUSION]")
    print(f"-> Found {len(harmful_nodes)} Noise-Generating Neurons (removing them improved network performance).")
    print("-> Top 3 CRITICAL Neurons (Network heavily relies on these):")
    for n_id, loss in critical_nodes[:3]:
        print(f"   Node [{n_id:02d}] caused {loss:.4f} drop when ablated.")

if __name__ == "__main__":
    main()
