import sys
import os
import torch
import numpy as np
import pandas as pd

# Set environment variable so load_data knows where to look
os.environ["PANEL_PATH"] = "Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv"

sys.path.append(os.path.abspath("Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs"))
import causal_cfm_cvae
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")

print("Loading data...")
X, Y, L, available_features, available_targets, norm_dict, treat_idx, cases, cell_assignments, filing_years = causal_cfm_cvae.load_data()

input_dim = X.shape[-1]
y_dim = Y.shape[-1]
prop_petition_idx = available_features.index("cumulative_petition_pct") if "cumulative_petition_pct" in available_features else None
skip_confounder_idx = None

val_ds = TensorDataset(X[:200], Y[:200], L[:200]) # just use a small subset for fast diagnostics
dl = DataLoader(val_ds, batch_size=32, shuffle=False)

Xb, Yb, Lb = next(iter(dl))
Xb, Yb = Xb.to(device), Yb.to(device)
B, T, _ = Yb.shape

model = causal_cfm_cvae.CausalSeq2SeqCFM(
    input_dim=input_dim,
    y_dim=y_dim,
    hidden_dim=128,
    latent_dim=32,
    treat_idx=treat_idx,
    prop_petition_idx=prop_petition_idx,
    skip_confounder_idx=skip_confounder_idx
).to(device)

model.eval()
with torch.no_grad():
    # Forward pass logic
    z, _, _ = model.encoder(Xb[:, :4, :])
    state   = model.transition.init_hidden(B, device)
    x_seq   = Xb[:, 4:, :]
    x_seq_u = x_seq.clone()
    for idx in treat_idx:
        x_seq_u[:, :, idx] = 0.0
    y_prev_seq = torch.cat([torch.zeros(B, 1, model.y_dim, device=device), Yb[:, 4:-1, :]], dim=1)
    inp_seq    = torch.cat([x_seq_u, y_prev_seq], dim=-1)
    out_seq, _ = model.transition.rnn(inp_seq, state)
    
    B_seq     = B * (T - 4)
    dose_flat = x_seq[:, :, treat_idx[0]].unsqueeze(-1).reshape(B_seq, 1)
    z_flat    = z.unsqueeze(1).repeat(1, T - 4, 1).reshape(B_seq, -1)
    h_flat    = out_seq.reshape(B_seq, model.hidden_dim)
    
    raw_parts = []
    if model.prop_petition_idx is not None:
        raw_parts.append(x_seq_u[:, :, model.prop_petition_idx].reshape(B_seq, -1))
    raw_conf = torch.cat(raw_parts, dim=-1) if raw_parts else None

    prop_logit, prop_mag = model.prop_net(z_flat, h_flat, raw_conf)
    prop_pred = torch.sigmoid(prop_logit) * prop_mag

    print("\n--- Diagnostic 2: Propensity ---")
    print(f"dose mean/std: {dose_flat.mean().item():.4f}, {dose_flat.std().item():.4f}")
    print(f"prop mean/std: {prop_pred.mean().item():.4f}, {prop_pred.std().item():.4f}")
    print(f"dose max: {dose_flat.max().item():.4f}, prop max: {prop_pred.max().item():.4f}")

    dose_enc = torch.cat([(dose_flat > 0).float(), dose_flat - torch.sigmoid(prop_logit) * prop_mag], dim=-1)
    gate_feat = torch.cat([z_flat, h_flat, dose_enc], dim=-1)
    gate_logit = model.height_gate(gate_feat)
    
    # 1 is the height concession pct target
    height_true = Yb[:, 4:, 1].reshape(B_seq, 1)

    print("\n--- Diagnostic 3: Hurdle Decomposition ---")
    print(f"true positive rate: {(height_true > 0).float().mean().item():.4f}")
    print(f"gate positive prob mean: {torch.sigmoid(gate_logit).mean().item():.4f}")
    
    pos_mask = height_true > 0
    zero_mask = height_true == 0
    if pos_mask.sum() > 0:
        print(f"gate prob on true positives: {torch.sigmoid(gate_logit)[pos_mask].mean().item():.4f}")
    else:
        print("gate prob on true positives: N/A")
        
    if zero_mask.sum() > 0:
        print(f"gate prob on true zeros: {torch.sigmoid(gate_logit)[zero_mask].mean().item():.4f}")
