import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score

import os
import causal_cfm_cvae
causal_cfm_cvae.PANEL_PATH = os.environ.get("PANEL_PATH", "biweekly_panel.csv")
from causal_cfm_cvae import load_data, CausalSeq2SeqCFM

MANIFEST_TEMPLATE = "output/feature_manifest_fold_{fold}.json"
CKPT_TEMPLATE     = "output/causal_cfm_weights_fold_{fold}.pt"

def load_manifest(fold: int) -> dict | None:
    """Load feature manifest if present. Returns None if not yet written (legacy checkpoints)."""
    path = MANIFEST_TEMPLATE.format(fold=fold)
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        print(f"  [manifest] Loaded: {path}")
        return m
    print(f"  [manifest] Not found ({path}) — falling back to weight-shape introspection.")
    return None

def validate_features(manifest: dict, local_features: list[str]) -> tuple[list[str], list[str]]:
    """Returns (missing_in_local, extra_in_local) relative to training manifest."""
    train_set  = set(manifest["feature_names"])
    local_set  = set(local_features)
    missing    = sorted(train_set - local_set)
    extra      = sorted(local_set - train_set)
    return missing, extra

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Evaluation Device: {device}\n", flush=True)

        X, Y, L, features, targets, norm_dict, treat_idx, cases, cell_assignments, filing_years = load_data()
        
        # Truncation block removed to support new AWS weights (55 features)
        
        n_cases   = len(X)
        input_dim = X.shape[-1]
        y_dim_actual = Y.shape[-1]


        f_cum_tok  = features.index("cumulative_council_nlp_lag1") if "cumulative_council_nlp_lag1" in features else None
        f_cum_comm = features.index("cumulative_commission_hearings_lag1") if "cumulative_commission_hearings_lag1" in features else None
        f_cum_coun = features.index("cumulative_council_hearings_lag1") if "cumulative_council_hearings_lag1" in features else None

        # ── Step 1: Load manifest (preferred) or introspect checkpoint ──────────
        fold = 0
        manifest = load_manifest(fold)

        print(f"Loading Double Machine Learning CFM weights (Fold {fold})...")
        raw_sd = torch.load(CKPT_TEMPLATE.format(fold=fold), map_location=device)
        raw_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
        
        # Legacy introspection
        prop_net_in = raw_sd['prop_net.shared.0.weight'].shape[1]
        rnn_hidden  = raw_sd['transition.rnn.weight_ih_l0'].shape[0] // 4
        skip_dim    = prop_net_in - rnn_hidden

        # The backfilled manifest has an incorrect skip_confounder_idx (length 50 instead of 7).
        # We explicitly override it here with the exact 7 features AWS used.
        skip_cols = [
            "dist_petition_rate_lag1", "knn_petition_rate_1km", 
            "race_white", "renter_share", "median_household_income", 
            "mortgage_rate_30yr_momentum", "fed_funds_rate_momentum"
        ]
        effective_skip = [features.index(c) for c in skip_cols if c in features]

        if manifest is not None and manifest.get("prop_petition_idx") is not None:
            petition_lag_idx = manifest["prop_petition_idx"]
        else:
            petition_lag_cols = [
                "cumulative_petition_pct_lag1",
                "cumulative_petition_count_lag1",
                "cumulative_petition_events_lag1",
                "petition_velocity_3p_lag1",
            ]
            petition_lag_idx = [features.index(c) for c in petition_lag_cols if c in features]

        treatment_derived_idx = (
            manifest.get("treatment_derived_idx", []) if manifest is not None else []
        )

        print(f"  Introspected: prop_net_in={prop_net_in}, rnn_hidden={rnn_hidden}, skip_dim={skip_dim}")
        print(f"  Using AWS training architecture: z_dim=64, skip_cols={len(effective_skip)}, petition_lags={len(petition_lag_idx)}")



        model = CausalSeq2SeqCFM(
            input_dim=input_dim, y_dim=y_dim_actual,
            hidden_dim=256, latent_dim=64,
            cfm_hidden=512, cfm_layers=5,
            n_layers=3, treat_idx=treat_idx,
            f_cum_tok=f_cum_tok, f_cum_comm=f_cum_comm, f_cum_coun=f_cum_coun,
            skip_confounder_idx=effective_skip,
            prop_petition_idx=petition_lag_idx,
            treatment_derived_idx=treatment_derived_idx,
            # prop_z_dim defaults to latent_dim=64 (re-enabled in fixed architecture)
        ).to(device)
        print(f"  Model: input_dim={input_dim}, skip_confounders={len(effective_skip)}, prop_net_in={256 + 64 + len(effective_skip)}")

        missing_keys, unexpected_keys = model.load_state_dict(raw_sd, strict=False)
        if missing_keys or unexpected_keys:
            print("  [WARN] Architecture/checkpoint mismatch. Do not trust eval until resolved.")
        if missing_keys:
            print(f"  [WARN] Missing ({len(missing_keys)}): {missing_keys[:3]}")
        if unexpected_keys:
            print(f"  [WARN] Unexpected ({len(unexpected_keys)}): {unexpected_keys[:3]}")
        print(f"  Weights loaded — {'CLEAN' if not missing_keys and not unexpected_keys else 'PARTIAL (strict=False)'}")
        model.eval()  # disable dropout for inference
        HT_IDX = 1  # height_concession_pct index in Y

        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), torch.tensor(L, dtype=torch.long))
        dl = DataLoader(ds, batch_size=128, shuffle=False)

        all_prop_preds, all_dose_true   = [], []
        all_ht_pred_pos, all_ht_true_pos = [], []
        all_ht_pred_zero, all_ht_true_zero = [], []
        all_gate_prob, all_gate_true = [], []

        has_hurdle = hasattr(model, 'height_gate') and hasattr(model, 'height_cfm')
        print(f"\nHurdle decoder present: {has_hurdle}")
        print("Running causal metric validation (split on concession > 0, not petition > 0)...")

        with torch.no_grad():
            for Xb, Yb, Lb in dl:
                Xb, Yb, Lb = Xb.to(device), Yb.to(device), Lb.to(device)
                B, T, _ = Xb.shape

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
                if model.skip_confounder_idx is not None:
                    raw_parts.append(x_seq_u[:, :, model.skip_confounder_idx].reshape(B_seq, -1))
                if model.prop_petition_idx is not None:
                    raw_parts.append(x_seq_u[:, :, model.prop_petition_idx].reshape(B_seq, -1))
                raw_conf = torch.cat(raw_parts, dim=-1) if raw_parts else None

                prop_logit, prop_mag = model.prop_net(z_flat, h_flat, raw_conf)
                prop_pred = torch.sigmoid(prop_logit) * prop_mag
                
                valid_t_prop = torch.arange(T - 4, device=device).unsqueeze(0)
                pad_mask_prop = (valid_t_prop < (Lb.unsqueeze(1) - 4).clamp(min=0)).reshape(-1).cpu().numpy()
                valid_rows_prop = pad_mask_prop > 0
                
                all_prop_preds.append(prop_pred.cpu().numpy()[valid_rows_prop])
                all_dose_true.append(dose_flat.cpu().numpy()[valid_rows_prop])

                # Hurdle-aware factual reconstruction. Requires sample(..., dose_val=None)
                preds_eval = model.sample(Xb[:, :4, :], Xb.clone(), dose_val=None, n_steps=10)

                # Gate diagnostics on valid rows.
                height_true_flat = Yb[:, 4:, HT_IDX:HT_IDX+1].reshape(B_seq, 1)
                dose_resid = dose_flat - prop_pred
                dose_enc_flat = torch.cat([(dose_flat > 0).float(), dose_resid], dim=-1)
                gate_logit = model.height_gate(torch.cat([z_flat, h_flat, dose_enc_flat], dim=-1))
                gate_prob = torch.sigmoid(gate_logit).detach().cpu().numpy().reshape(-1)
                gate_true = (height_true_flat.detach().cpu().numpy().reshape(-1) > 0).astype(int)
                all_gate_prob.append(gate_prob[valid_rows_prop])
                all_gate_true.append(gate_true[valid_rows_prop])

                ht_pred = preds_eval[:, 4:, HT_IDX].cpu().numpy().flatten()
                ht_true = Yb[:, 4:, HT_IDX].cpu().numpy().flatten()
                
                valid_t = torch.arange(T - 4, device=device).unsqueeze(0)
                pad_mask = (valid_t < (Lb.unsqueeze(1) - 4).clamp(min=0)).reshape(-1).cpu().numpy()
                
                valid_rows = pad_mask > 0
                ht_pred = ht_pred[valid_rows]
                ht_true = ht_true[valid_rows]

                pos_mask  = ht_true > 0
                zero_mask = ~pos_mask

                all_ht_pred_pos.append(ht_pred[pos_mask])
                all_ht_true_pos.append(ht_true[pos_mask])
                all_ht_pred_zero.append(ht_pred[zero_mask])
                all_ht_true_zero.append(ht_true[zero_mask])

        prop_preds_np = np.concatenate(all_prop_preds)
        dose_true_np  = np.concatenate(all_dose_true)
        ht_pred_pos   = np.concatenate(all_ht_pred_pos)  if all_ht_pred_pos  else np.array([])
        ht_true_pos   = np.concatenate(all_ht_true_pos)  if all_ht_true_pos  else np.array([])
        ht_pred_zero  = np.concatenate(all_ht_pred_zero) if all_ht_pred_zero else np.array([])
        ht_true_zero  = np.concatenate(all_ht_true_zero) if all_ht_true_zero else np.array([])

        prop_mse = np.mean((prop_preds_np - dose_true_np)**2)
        prop_r2  = r2_score(dose_true_np, prop_preds_np)
        print("\n" + "="*60)
        print("PROPENSITY NETWORK (DML ORTHOGONALIZATION)")
        print("="*60)
        print(f"  Propensity MSE : {prop_mse:.4f}")
        print(f"  Propensity R\u00b2  : {prop_r2:.4f}")
        print(f"  dose mean/std/max: {dose_true_np.mean():.6f} / {dose_true_np.std():.6f} / {dose_true_np.max():.6f}")
        print(f"  prop mean/std/max: {prop_preds_np.mean():.6f} / {prop_preds_np.std():.6f} / {prop_preds_np.max():.6f}")
        if prop_r2 > 0.1:
            print("  -> Baseline variance captured. Dose residual orthogonalized.")
        else:
            print("  -> FAIL: PropNet predicts constant. No DML effect.")

        n_pos  = len(ht_true_pos)
        n_zero = len(ht_true_zero)
        n_tot  = n_pos + n_zero
        print("\n" + "="*60)
        print("HURDLE HEIGHT DECODER (split on concession > 0, not petition > 0)")
        print("="*60)
        print(f"  Concession==0 rows : {n_zero:,}  ({100*n_zero/max(n_tot,1):.1f}%)")
        print(f"  Concession >0 rows : {n_pos:,}   ({100*n_pos/max(n_tot,1):.1f}%)")
        if n_zero > 0:
            mae_zero = np.mean(np.abs(ht_pred_zero - ht_true_zero))
            print(f"  MAE | concession==0 : {mae_zero:.4f}  (target: near 0)")
        if n_pos > 0:
            mae_pos = np.mean(np.abs(ht_pred_pos - ht_true_pos))
            print(f"  MAE | concession >0 : {mae_pos:.4f}  (pass threshold: < 0.20)")
            if mae_pos < 0.20:
                print("  -> SUCCESS: Conditional CFM learned the positive-concession manifold.")
            else:
                print("  -> FAIL: Conditional CFM still imprecise on positive-concession rows.")

        gate_prob_np = np.concatenate(all_gate_prob)
        gate_true_np = np.concatenate(all_gate_true)
        print("\nHEIGHT GATE")
        print(f"  true positive concession rate: {gate_true_np.mean():.4f}")
        print(f"  gate mean: {gate_prob_np.mean():.4f}")
        if gate_true_np.sum() > 0:
            print(f"  gate mean on true positives: {gate_prob_np[gate_true_np == 1].mean():.4f}")
        if (gate_true_np == 0).sum() > 0:
            print(f"  gate mean on true zeros: {gate_prob_np[gate_true_np == 0].mean():.4f}")
        if len(np.unique(gate_true_np)) == 2:
            print(f"  gate AUROC: {roc_auc_score(gate_true_np, gate_prob_np):.4f}")
            print(f"  gate AP:    {average_precision_score(gate_true_np, gate_prob_np):.4f}")
        else:
            print("  gate AUROC/AP: undefined; only one class present.")
        print("="*60 + "\n")

    except Exception as e:
        import traceback
        print(f"\nFATAL ERROR:\n{traceback.format_exc()}", flush=True)

if __name__ == "__main__":
    main()
