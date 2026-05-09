"""
causal_cfm_inference.py -- G-computation causal surfaces from saved CFM folds.

Key fixes over the old inference script:
  - Reads norm_dict from the fold manifest; does NOT refit a StandardScaler
  - Petition dose left in raw [0, 1] -- matches training normalization
  - Imposes counterfactual via X_cf[:, :, treat_idx]; dose_val=None so the
    model reads dose from X_cf at each period (correct G-computation)
  - Uses MEAN expected concession, not p50 median (which is always 0 for an
    85%-zero-inflated target)
  - height output is already E[Y] = P(Y>0) * E[Y|Y>0] from model.sample()

Usage:
  python causal_cfm_inference.py [--folds 0,1,2,3,4] [--n_steps 20]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# Allow import of model from sibling file
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from causal_cfm_cvae import (
    CausalSeq2SeqCFM,
    HEIGHT_EPS,
    HEIGHT_TARGET,
    T_MAX,
    PRE_PERIODS,
    device,
    build_tensors,
)

OUT_DIR = os.environ.get("OUT_DIR", ".")
PANEL_PATH = os.environ.get(
    "PANEL_PATH",
    os.path.join(OUT_DIR, "biweekly_panel.csv"),
)

INTENSITIES = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
TIMINGS = [5, 9, 13]        # period indices (1-based) to inject shock


# ── Normalization helpers ─────────────────────────────────────────────────────

def apply_manifest_norm(df: pd.DataFrame, norm_dict: dict) -> pd.DataFrame:
    """
    Apply exactly the same normalization used during training.
    Petition columns and height_concession_pct are intentionally absent from
    norm_dict and stay in raw units.
    """
    df = df.copy()
    for col, spec in norm_dict.items():
        if col.startswith("_"):
            continue
        if col not in df.columns:
            continue
        if isinstance(spec, list):
            m, s = float(spec[0]), float(spec[1])
        else:
            continue  # scalar entries are stats, not feature norms
        df[col] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0.0) - m
        ) / max(s, 1e-8)
    return df


def decode_preds(preds: np.ndarray, targets: list, norm_dict: dict) -> np.ndarray:
    """
    Inverse-transform model outputs back to interpretable units.

    preds : (B, T, Y_DIM)
    Returns same shape array with:
      - resolved      : probability in [0, 1]
      - height_pct    : clipped to [0, 1]
      - cont targets  : un-z-scored to original scale
    """
    out = preds.copy()
    for j, target in enumerate(targets):
        if target == "resolved":
            out[..., j] = 1.0 / (1.0 + np.exp(-out[..., j]))
        elif target == HEIGHT_TARGET:
            out[..., j] = np.clip(out[..., j], 0.0, 1.0)
        elif target in norm_dict:
            spec = norm_dict[target]
            if isinstance(spec, list):
                m, s = float(spec[0]), float(spec[1])
                out[..., j] = out[..., j] * s + m
    return out


# ── Model loading ─────────────────────────────────────────────────────────────

def load_fold_model(fold: int) -> "tuple[CausalSeq2SeqCFM, dict]":
    mf_path = os.path.join(OUT_DIR, f"feature_manifest_fold_{fold}.json")
    ckpt_path = os.path.join(OUT_DIR, f"cfm_weights_fold_{fold}.pt")

    if not os.path.exists(mf_path):
        raise FileNotFoundError(f"Manifest not found: {mf_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, manifest


# ── Core inference ────────────────────────────────────────────────────────────

def run_fold_inference(
    fold: int,
    df_raw: pd.DataFrame,
    n_steps: int = 20,
    height_mc_samples: int = 8,
) -> pd.DataFrame:
    """
    Run counterfactual G-computation for one fold.
    Returns a long-form DataFrame with columns:
      fold, case_number, timing, intensity, target, mean, p10, p90
    """
    model, manifest = load_fold_model(fold)
    features = manifest["features"]
    targets = manifest["targets"]
    norm_dict = manifest["norm_dict"]
    treat_idx = manifest["treat_idx"]
    height_idx = manifest.get("height_idx", targets.index(HEIGHT_TARGET) if HEIGHT_TARGET in targets else 1)

    df = apply_manifest_norm(df_raw, norm_dict)

    # Use all cases available (ensemble across folds already averages across train/test splits)
    all_cases = df["case_number"].unique()
    X_all, Y_all, L_all = build_tensors(df, features, targets, all_cases)

    print(
        f"  [fold {fold}] {len(all_cases)} cases | "
        f"shape={tuple(X_all.shape)} | device={device}",
        flush=True,
    )

    rows = []

    for timing in TIMINGS:
        for intensity in INTENSITIES:
            # Build counterfactual tensor: single shock at `timing`, cumulative persists
            X_cf = X_all.clone().to(device)
            # Zero out all existing treatment variation
            X_cf[:, :, treat_idx[0]] = 0.0
            if len(treat_idx) > 1:
                X_cf[:, :, treat_idx[1]] = 0.0

            # Inject shock at `timing` (period index, 1-based -> 0-based: timing-1)
            shock_t = timing - 1
            X_cf[:, shock_t, treat_idx[0]] = float(intensity)
            if len(treat_idx) > 1:
                # Cumulative petition persists from shock forward
                X_cf[:, shock_t:, treat_idx[1]] = float(intensity)

            X_pre = X_cf[:, :PRE_PERIODS, :]

            ensemble_preds = []
            with torch.no_grad():
                preds = model.sample(
                    X_pre,
                    X_cf.clone(),
                    dose_val=None,           # use per-period do-policy in X_cf
                    n_steps=n_steps,
                    height_mc_samples=height_mc_samples,
                    sample_height_atom=False,  # expected value, not Bernoulli draw
                )
                ensemble_preds.append(preds.cpu().numpy())

            avg_preds = np.mean(ensemble_preds, axis=0)  # (B, T, Y)
            decoded = decode_preds(avg_preds, targets, norm_dict)

            # Terminal-period outcomes (last observed period per case)
            L_np = L_all.numpy()
            for j, target in enumerate(targets):
                # Gather terminal value for each case
                vals = np.array([
                    decoded[i, max(PRE_PERIODS, int(L_np[i]) - 1), j]
                    for i in range(len(all_cases))
                ])
                rows.append({
                    "fold": fold,
                    "timing": timing,
                    "intensity": intensity,
                    "target": target,
                    "mean": float(np.mean(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                    "n_cases": len(vals),
                })

    return pd.DataFrame(rows)


# ── ATE surface builder ───────────────────────────────────────────────────────

def build_ate_surface(results_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Compute ATE = E[Y(d)] - E[Y(0)] for each (timing, intensity) cell,
    averaged across folds.
    """
    sub = results_df[results_df["target"] == target].copy()
    base = sub[sub["intensity"] == 0.0][["fold", "timing", "mean"]].rename(
        columns={"mean": "baseline"}
    )
    sub = sub.merge(base, on=["fold", "timing"], how="left")
    sub["ate"] = sub["mean"] - sub["baseline"]

    surface = (
        sub.groupby(["timing", "intensity"])
        .agg(
            ate_mean=("ate", "mean"),
            ate_std=("ate", "std"),
            mean_y=("mean", "mean"),
        )
        .reset_index()
    )
    surface["target"] = target
    return surface


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folds", type=str, default="0,1,2,3,4",
        help="Comma-separated fold indices to ensemble"
    )
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--height_mc", type=int, default=8)
    args = parser.parse_args()

    fold_list = [int(f) for f in args.folds.split(",")]

    print("=" * 60, flush=True)
    print(f"Causal CFM Inference  |  folds={fold_list}  |  device={device}", flush=True)
    print("=" * 60, flush=True)

    # Load raw panel once (normalization applied per-fold inside run_fold_inference)
    if os.path.exists("/home/ubuntu/biweekly_panel.csv"):
        df_raw = pd.read_csv("/home/ubuntu/biweekly_panel.csv", low_memory=False)
    else:
        df_raw = pd.read_csv(PANEL_PATH, low_memory=False)

    df_raw["period_start_dt"] = pd.to_datetime(df_raw["period_start"], errors="coerce")
    df_raw = df_raw.sort_values(["case_number", "period_seq"]).reset_index(drop=True)

    all_results = []
    for fold in fold_list:
        print(f"\n--- Fold {fold} ---", flush=True)
        try:
            fold_df = run_fold_inference(
                fold, df_raw,
                n_steps=args.n_steps,
                height_mc_samples=args.height_mc,
            )
            all_results.append(fold_df)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", flush=True)

    if not all_results:
        print("No fold results -- exiting.", flush=True)
        return

    combined = pd.concat(all_results, ignore_index=True)

    # ── G-computation ATE surfaces ────────────────────────────────────────────
    surfaces = []
    for target in combined["target"].unique():
        surf = build_ate_surface(combined, target)
        surfaces.append(surf)
        if target == HEIGHT_TARGET:
            print(f"\n--- ATE Surface: {target} ---", flush=True)
            pivot = surf.pivot(index="timing", columns="intensity", values="ate_mean").round(4)
            print(pivot.to_string(), flush=True)

    surface_df = pd.concat(surfaces, ignore_index=True)

    # Save outputs
    raw_path = os.path.join(OUT_DIR, "cfm_counterfactual_raw.csv")
    surf_path = os.path.join(OUT_DIR, "cfm_ate_surfaces.csv")
    combined.to_csv(raw_path, index=False)
    surface_df.to_csv(surf_path, index=False)

    print(f"\n[OK] Raw results: {raw_path}", flush=True)
    print(f"[OK] ATE surfaces: {surf_path}", flush=True)


if __name__ == "__main__":
    main()
