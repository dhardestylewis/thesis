"""
evaluate_causal_fixes.py -- Hurdle-specific diagnostics and G-computation ATE sanity check.

Validates that the Zero-Inflated Causal Hurdle CFM:
  1. Correctly predicts the height concession event gate (not just nonzero petition dose)
  2. Produces a non-zero G-computation ATE for height under do(1) vs do(0)
  3. Covers the positive-concession manifold (not just the dose manifold)

Usage:
  python evaluate_causal_fixes.py --fold 0 [--n_cases 200]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from causal_cfm_cvae import (
    CausalSeq2SeqCFM,
    HEIGHT_EPS,
    HEIGHT_TARGET,
    PRE_PERIODS,
    T_MAX,
    device,
    load_data,
    build_tensors,
)

OUT_DIR = os.environ.get("OUT_DIR", ".")


def load_model_and_manifest(fold: int) -> "tuple[CausalSeq2SeqCFM, dict]":
    mf_path = os.path.join(OUT_DIR, f"feature_manifest_fold_{fold}.json")
    ck_path = os.path.join(OUT_DIR, f"cfm_weights_fold_{fold}.pt")

    with open(mf_path) as fh:
        manifest = json.load(fh)

    norm_dict = manifest["norm_dict"]

    model = CausalSeq2SeqCFM(
        x_dim=manifest["x_dim"],
        y_dim=manifest["y_dim"],
        treat_idx=manifest["treat_idx"],
        skip_confounder_idx=manifest.get("skip_confounder_idx") or None,
        resolved_idx=manifest.get("resolved_idx", 0),
        height_idx=manifest.get("height_idx", 1),
        f_cum_tok=manifest.get("f_cum_tok"),
        t_idx_tok=manifest.get("t_idx_tok"),
        f_cum_comm=manifest.get("f_cum_comm"),
        t_idx_comm=manifest.get("t_idx_comm"),
        f_cum_coun=manifest.get("f_cum_coun"),
        t_idx_coun=manifest.get("t_idx_coun"),
    ).to(device)

    model.height_u_mean.fill_(float(norm_dict.get("_height_pos_logit_mean", 0.0)))
    model.height_u_std.fill_(float(norm_dict.get("_height_pos_logit_std", 1.0)))
    model.height_pos_weight.fill_(float(norm_dict.get("_height_pos_weight", 10.0)))

    model.load_state_dict(torch.load(ck_path, map_location=device))
    model.eval()
    return model, manifest


# ── Diagnostic 1: Hurdle gate performance ────────────────────────────────────

def eval_hurdle_gate(model, X, Y, L, targets, batch_size=64):
    """
    Evaluate the height event gate on TRUE positive-concession periods
    (not on nonzero-petition periods, which is a different event).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    height_idx = model.height_idx

    all_p_pos, all_label, all_valid = [], [], []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = X[i: i + batch_size].to(device)
            Yb = Y[i: i + batch_size].to(device)
            Lb = L[i: i + batch_size].to(device)

            B, T, _ = Xb.shape
            n_pred = T - PRE_PERIODS

            steps = torch.arange(PRE_PERIODS, T, device=device).unsqueeze(0)
            valid = steps < Lb.unsqueeze(1)   # (B, n_pred)

            z, mu, lv = model.encoder(Xb[:, :PRE_PERIODS, :])
            y_prev = torch.cat([
                torch.zeros(B, 1, model.y_dim, device=device),
                Yb[:, PRE_PERIODS:-1, :],
            ], dim=1)
            x_unb = Xb[:, PRE_PERIODS:, :].clone()
            for idx in model.treat_idx:
                x_unb[:, :, idx] = 0.0

            state = model.transition.init_hidden(B, device)
            inp = torch.cat([x_unb, y_prev], dim=-1)
            out_seq, _ = model.transition.rnn(inp, state)

            B_seq = B * n_pred
            dose_flat = Xb[:, PRE_PERIODS:, model.treat_idx[0]].unsqueeze(-1).reshape(B_seq, 1)
            z_flat = z.unsqueeze(1).repeat(1, n_pred, 1).reshape(B_seq, -1)
            h_flat = out_seq.reshape(B_seq, model.hidden_dim)

            raw_cf = None
            if model.skip_confounder_idx:
                raw_cf = x_unb[:, :, model.skip_confounder_idx].reshape(B_seq, -1)

            prop_logit, prop_mag = model.prop_net(z_flat, h_flat, raw_cf)
            dose_nz = (dose_flat > 0).float()
            dose_residual = dose_flat - (torch.sigmoid(prop_logit) * prop_mag)
            dose_enc = torch.cat([dose_nz, dose_residual], dim=-1)

            gate_logit = model.height_gate(z_flat, h_flat, dose_enc)
            p_pos = torch.sigmoid(gate_logit).reshape(B, n_pred)

            height_true = Yb[:, PRE_PERIODS:, height_idx]   # (B, n_pred)
            label = (height_true > HEIGHT_EPS).float()

            all_p_pos.append(p_pos.cpu())
            all_label.append(label.cpu())
            all_valid.append(valid.cpu())

    p_all = torch.cat(all_p_pos, dim=0).reshape(-1).numpy()
    l_all = torch.cat(all_label, dim=0).reshape(-1).numpy()
    v_all = torch.cat(all_valid, dim=0).reshape(-1).numpy().astype(bool)

    p_v = p_all[v_all]
    l_v = l_all[v_all]

    pos_true_rate = float(l_v.mean())
    pos_pred_rate = float((p_v > 0.5).mean())
    mae_gate = float(np.abs(p_v - l_v).mean())

    if l_v.sum() > 5 and len(np.unique(l_v)) > 1:
        roc = roc_auc_score(l_v, p_v)
        pr = average_precision_score(l_v, p_v)
    else:
        roc = pr = float("nan")

    print("\n=== HURDLE GATE DIAGNOSTICS ===", flush=True)
    print(f"  Valid timesteps evaluated : {int(v_all.sum())}", flush=True)
    print(f"  True positive concession rate  : {pos_true_rate:.4f}", flush=True)
    print(f"  Pred positive concession rate  : {pos_pred_rate:.4f}", flush=True)
    print(f"  Gate MAE                       : {mae_gate:.4f}", flush=True)
    print(f"  Gate ROC-AUC                   : {roc:.4f}", flush=True)
    print(f"  Gate PR-AUC                    : {pr:.4f}", flush=True)

    # Comparison: nonzero petition rate (should differ from concession rate)
    if len(X) > 0:
        dose_col_idx = model.treat_idx[0]
        dose_all = X[:, PRE_PERIODS:, dose_col_idx].numpy()
        L_np = L.numpy()
        mask2d = np.arange(T_MAX - PRE_PERIODS)[None, :] < (L_np[:, None] - PRE_PERIODS)
        dose_nz_rate = float((dose_all[mask2d] > 0).mean())
        print(f"  (Nonzero petition rate for comparison : {dose_nz_rate:.4f})", flush=True)
        if abs(dose_nz_rate - pos_true_rate) < 0.01:
            print(
                "  WARNING: Concession rate ≈ petition rate -- these may be confounded.",
                flush=True,
            )


# ── Diagnostic 2: G-computation ATE sanity check ─────────────────────────────

def gcomp_ate_sanity(model, X, Y, L, targets, n_cases=200):
    """
    Compute E[Y(1)] - E[Y(0)] via G-computation.
    For a valid causal model, this should be POSITIVE and non-trivially large
    when the dose causes height concessions.
    """
    height_idx = model.height_idx
    targets_list = targets

    n = min(n_cases, len(X))
    Xb = X[:n].to(device)

    # do(0): zero petition everywhere
    X0 = Xb.clone()
    X0[:, :, model.treat_idx[0]] = 0.0
    if len(model.treat_idx) > 1:
        X0[:, :, model.treat_idx[1]] = 0.0

    # do(1): petition=1.0 from period 5 onward (shock at t=4, cumulative from t=4)
    X1 = Xb.clone()
    X1[:, :, model.treat_idx[0]] = 0.0
    if len(model.treat_idx) > 1:
        X1[:, :, model.treat_idx[1]] = 0.0
    X1[:, PRE_PERIODS, model.treat_idx[0]] = 1.0
    if len(model.treat_idx) > 1:
        X1[:, PRE_PERIODS:, model.treat_idx[1]] = 1.0

    model.eval()
    with torch.no_grad():
        pred0 = model.sample(X0[:, :PRE_PERIODS, :], X0.clone(), dose_val=None, sample_height_atom=False)
        pred1 = model.sample(X1[:, :PRE_PERIODS, :], X1.clone(), dose_val=None, sample_height_atom=False)

    p0 = pred0.cpu().numpy()
    p1 = pred1.cpu().numpy()

    # Terminal predictions
    L_np = L[:n].numpy()
    t_idx = np.minimum(L_np - 1, T_MAX - 1).clip(min=PRE_PERIODS)

    h0 = np.array([p0[i, t_idx[i], height_idx] for i in range(n)])
    h1 = np.array([p1[i, t_idx[i], height_idx] for i in range(n)])

    ate = float((h1 - h0).mean())
    ate_std = float((h1 - h0).std())

    print("\n=== G-COMPUTATION ATE SANITY CHECK ===", flush=True)
    print(f"  Cases evaluated    : {n}", flush=True)
    print(f"  E[Y(0)] height     : {h0.mean():.6f}", flush=True)
    print(f"  E[Y(1)] height     : {h1.mean():.6f}", flush=True)
    print(f"  ATE = E[Y(1)-Y(0)] : {ate:+.6f}  (std={ate_std:.6f})", flush=True)

    if abs(ate) < 1e-5:
        print(
            "  !! ATE is effectively zero -- height causal signal may still be collapsed.",
            flush=True,
        )
    elif ate < 0:
        print(
            "  !! ATE is negative -- check treatment direction and sign conventions.",
            flush=True,
        )
    else:
        print(
            "  OK: ATE is positive -- height concession responds to petition dose.",
            flush=True,
        )

    # Check other targets too
    for j, tgt in enumerate(targets_list):
        if tgt in (HEIGHT_TARGET, "resolved"):
            continue
        y0 = np.array([p0[i, t_idx[i], j] for i in range(n)])
        y1 = np.array([p1[i, t_idx[i], j] for i in range(n)])
        delta = float((y1 - y0).mean())
        print(f"  ATE [{tgt:35s}]: {delta:+.4f}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_cases", type=int, default=200,
                        help="Cases for G-computation ATE check")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"Hurdle CFM Diagnostics  |  fold={args.fold}  |  device={device}", flush=True)
    print("=" * 60, flush=True)

    print("Loading model and manifest...", flush=True)
    model, manifest = load_model_and_manifest(args.fold)
    features = manifest["features"]
    targets = manifest["targets"]

    print("Loading and normalizing data...", flush=True)
    df, _, _, norm_dict = load_data()

    all_cases = df["case_number"].unique()
    X, Y, L = build_tensors(df, features, targets, all_cases)

    print(f"  Loaded {len(all_cases)} cases, tensor shape {tuple(X.shape)}", flush=True)

    eval_hurdle_gate(model, X, Y, L, targets)
    gcomp_ate_sanity(model, X, Y, L, targets, n_cases=args.n_cases)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
