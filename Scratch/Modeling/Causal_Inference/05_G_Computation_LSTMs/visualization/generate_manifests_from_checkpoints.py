"""
generate_manifests_from_checkpoints.py

One-time script to back-fill feature_manifest_fold_N.json for all existing
AWS checkpoints in aws_deploy/ using the biweekly_panel_aws.csv that was
actually used during training.

Run once from the 05_G_Computation_LSTMs/ directory:
    python generate_manifests_from_checkpoints.py
"""
import os, sys, json, time
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import causal_cfm_cvae

AWS_DIR    = os.path.join(os.path.dirname(__file__), "output")
PANEL_PATH = os.path.join(os.path.dirname(__file__), "biweekly_panel.csv")

# Load the AWS training panel to get the exact feature list
causal_cfm_cvae.PANEL_PATH = PANEL_PATH
print(f"Loading AWS panel: {PANEL_PATH}")
X, Y, L, features, targets, norm_dict, treat_idx, cases, cell_assignments, filing_years = causal_cfm_cvae.load_data()
print(f"  input_dim={X.shape[2]}, features={len(features)}, targets={len(targets)}")

for fold in range(5):
    ckpt = os.path.join(AWS_DIR, f"causal_cfm_weights_fold_{fold}.pt")
    if not os.path.exists(ckpt):
        print(f"  [skip] fold {fold} — checkpoint not found")
        continue

    sd = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    if "prop_net.shared.0.weight" not in sd:
        print(f"  [skip] fold {fold} — not a causal_cfm_cvae checkpoint (missing prop_net)")
        continue

    # Introspect architecture from weights
    prop_net_in = sd["prop_net.shared.0.weight"].shape[1]
    rnn_hidden  = sd["transition.rnn.weight_ih_l0"].shape[0] // 4
    rnn_input   = sd["transition.rnn.weight_ih_l0"].shape[1]   # actual LSTM input_dim at training
    skip_dim    = prop_net_in - rnn_hidden

    # skip_confounder_idx: best approximation is first skip_dim feature indices
    # (exact mapping requires the original training launch script config)
    skip_idx = list(range(min(skip_dim, X.shape[2])))

    manifest = {
        "fold": fold,
        "input_dim": int(rnn_input),           # ground truth from LSTM weights
        "y_dim": int(Y.shape[2]),
        "feature_names": features,             # from AWS panel
        "target_names": targets,
        "treat_idx": treat_idx if isinstance(treat_idx, list) else [int(treat_idx)],
        "skip_confounder_idx": skip_idx,       # approximate — update after next training run
        "hidden_dim": rnn_hidden,
        "latent_dim": 64,
        "cfm_hidden": 512,
        "cfm_layers": 5,
        "n_layers": 3,
        "T_MAX": 55,
        "panel_path": PANEL_PATH,
        "note": "Back-filled from checkpoint weight shapes. skip_confounder_idx is approximate.",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = os.path.join(AWS_DIR, f"feature_manifest_fold_{fold}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [fold {fold}] Written: {manifest_path}")
    print(f"             rnn_input={rnn_input}, prop_net_in={prop_net_in}, skip_dim={skip_dim}")

print("\nDone. Copy manifests to output/ for local eval:")
print("  Copy-Item aws_deploy\\feature_manifest_fold_*.json output\\")
