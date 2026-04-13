"""
irm_vrex_first_result.py — IRM vs V-REx vs ERM on Austin zoning environments
=============================================================================
First empirical test of the invariance premise.

Target:    log(total_market_value)
Features:  year_built, improvement_sq_ft, deed_acreage, land_market_value,
           homesite_flag, year (panel year)
Environments: 244 closed multi-parcel zoning events (2018–2025)

Models:
  1. ERM  — pooled OLS (ordinary least-squares)
  2. IRM  — OLS + gradient norm penalty (Arjovsky et al. 2019)
  3. V-REx — OLS + variance of per-environment risks (Krueger et al. 2021)

Evaluation:
  - Mean risk across environments
  - Worst-case (max) risk across environments
  - Risk gap = max(R^e) - min(R^e)  (should shrink under IRM/V-REx)

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "Analysis", "Results")

# Features to use (available in panel)
# NOTE: improvement_sq_ft is actually a building-class code (string), not numeric.
# homesite_flag is 'Y'/'N' string. improvement_market_value/appraised_value are
# mostly NaN in this panel — using them drops 98% of data.
FEATURE_COLS = [
    'year_built', 'deed_acreage',
    'land_market_value', 'year',
    'land_acres', 'new_construction_value',
]
TARGET_COL = 'total_market_value'
MIN_ENV_SIZE = 5  # minimum parcels per environment for inclusion


def load_data():
    """Load panel + environment assignments, merge, clean."""
    print("Loading environment assignments...")
    env = pd.read_csv(ENV_PATH)
    env = env.rename(columns={
        'irm_environment_id': 'env_id',
        'irm_environment_type': 'env_type'
    }) if 'irm_environment_id' in env.columns else env.rename(columns={
        'CASE_NUMBER': 'env_id',
        'SUB_TYPE': 'env_type'
    })
    print(f"  {len(env)} parcel-environment assignments")

    print("Loading panel (selective columns)...")
    usecols = ['standardized_tcad_id', 'year'] + FEATURE_COLS + [TARGET_COL]
    # Deduplicate in case TARGET_COL or year is in FEATURE_COLS
    usecols = list(dict.fromkeys(usecols))
    panel = pd.read_csv(PANEL_PATH, usecols=usecols, low_memory=False)
    panel = panel[panel['year'].between(2018, 2025)]
    print(f"  Panel 2018-2025: {panel.shape}")

    # Merge
    df = panel.merge(env, on='standardized_tcad_id', how='inner')
    print(f"  After merge (treated parcels only): {df.shape}")

    # Clean target
    df = df[df[TARGET_COL] > 0].copy()
    df['log_tmv'] = np.log(df[TARGET_COL])

    # Clean features
    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # new_construction_value: NaN means no new construction → fill with 0
    df['new_construction_value'] = df['new_construction_value'].fillna(0)
    df = df.dropna(subset=FEATURE_COLS + ['log_tmv'])
    print(f"  After cleaning: {df.shape}")

    # Filter environments with >= MIN_ENV_SIZE observations
    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]
    print(f"  After env size filter (>={MIN_ENV_SIZE}): {df.shape}")
    print(f"  Valid environments: {len(valid_envs)}")

    return df


def compute_env_risks(y_true, y_pred, env_ids):
    """Compute per-environment MSE."""
    risks = {}
    for eid in np.unique(env_ids):
        mask = env_ids == eid
        if mask.sum() > 0:
            risks[eid] = np.mean((y_true[mask] - y_pred[mask]) ** 2)
    return risks


def run_erm(X_train, y_train, X_test, y_test, env_train, env_test):
    """Standard pooled Ridge regression (ERM baseline)."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_risks = compute_env_risks(y_train, y_pred_train, env_train)
    test_risks = compute_env_risks(y_test, y_pred_test, env_test)

    return model, train_risks, test_risks


def irm_penalty_linear(X, y, w):
    """IRM gradient penalty for linear model: ||grad_{w=1} R^e(w*Phi)||^2
    For a linear model Phi(x) = Xw, the IRM penalty per environment is
    the squared norm of the gradient of the loss w.r.t. a scalar multiplier w=1.
    """
    residuals = y - X @ w
    grad = -2 * np.mean(residuals[:, None] * X, axis=0)
    return np.sum(grad ** 2)


def run_irm(X_train, y_train, env_train, X_test, y_test, env_test,
            lam=1e2, lr=0.001, n_iters=2000):
    """IRM via gradient descent on linear model."""
    n_features = X_train.shape[1]
    w = np.zeros(n_features)
    unique_envs = np.unique(env_train)

    for it in range(n_iters):
        # ERM gradient (pooled MSE)
        residuals = y_train - X_train @ w
        erm_grad = -2 * (X_train.T @ residuals) / len(y_train)

        # IRM penalty gradient
        irm_grad = np.zeros_like(w)
        for eid in unique_envs:
            if np.sum(env_train == eid) < 2:
                continue
            mask = env_train == eid
            Xe, ye = X_train[mask], y_train[mask]
            res_e = ye - Xe @ w
            # Gradient of ||grad_w R^e||^2 w.r.t. w
            grad_e = -2 * np.mean(res_e[:, None] * Xe, axis=0)
            # Chain rule: d/dw ||grad_e||^2 = 2 * (d grad_e / dw)^T * grad_e
            # d grad_e / dw = 2 * X^T X / n_e
            hess_e = 2 * (Xe.T @ Xe) / len(ye)
            irm_grad += 2 * hess_e @ grad_e / len(unique_envs)

        total_grad = erm_grad + lam * irm_grad
        # Gradient clipping
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm > 10.0:
            total_grad = total_grad * 10.0 / grad_norm
        w -= lr * total_grad

        if it % 500 == 0:
            loss = np.mean(residuals ** 2)
            pen = np.mean([irm_penalty_linear(X_train[env_train == e], y_train[env_train == e], w)
                           for e in unique_envs])
            print(f"    IRM iter {it:4d}: MSE={loss:.4f}  penalty={pen:.6f}")

    y_pred_train = X_train @ w
    y_pred_test = X_test @ w
    train_risks = compute_env_risks(y_train, y_pred_train, env_train)
    test_risks = compute_env_risks(y_test, y_pred_test, env_test)
    return w, train_risks, test_risks


def run_vrex(X_train, y_train, env_train, X_test, y_test, env_test,
             beta=1e2, lr=0.001, n_iters=2000):
    """V-REx: penalize variance of per-environment risks."""
    n_features = X_train.shape[1]
    w = np.zeros(n_features)
    unique_envs = np.unique(env_train)

    for it in range(n_iters):
        # Compute per-env risks and their variance
        env_risks = []
        env_grads = []
        for eid in unique_envs:
            mask = env_train == eid
            Xe, ye = X_train[mask], y_train[mask]
            res_e = ye - Xe @ w
            risk_e = np.mean(res_e ** 2)
            grad_e = -2 * (Xe.T @ res_e) / len(ye)
            env_risks.append(risk_e)
            env_grads.append(grad_e)

        env_risks = np.array(env_risks)
        mean_risk = env_risks.mean()

        # ERM gradient (mean of env gradients)
        erm_grad = np.mean(env_grads, axis=0)

        # V-REx gradient: d/dw Var(R^e) = d/dw (1/|E| sum (R^e - mean)^2)
        #   = 2/|E| sum (R^e - mean) * dR^e/dw
        vrex_grad = np.zeros_like(w)
        for i, eid in enumerate(unique_envs):
            vrex_grad += 2 * (env_risks[i] - mean_risk) * env_grads[i] / len(unique_envs)

        total_grad = erm_grad + beta * vrex_grad
        # Gradient clipping
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm > 10.0:
            total_grad = total_grad * 10.0 / grad_norm
        w -= lr * total_grad

        if it % 500 == 0:
            print(f"    V-REx iter {it:4d}: mean_risk={mean_risk:.4f}  Var(R^e)={env_risks.var():.6f}")

    y_pred_train = X_train @ w
    y_pred_test = X_test @ w
    train_risks = compute_env_risks(y_train, y_pred_train, env_train)
    test_risks = compute_env_risks(y_test, y_pred_test, env_test)
    return w, train_risks, test_risks


def summarize_risks(risks, label):
    """Print risk summary for a method."""
    vals = np.array(list(risks.values()))
    print(f"  {label}:")
    print(f"    Mean risk:      {vals.mean():.4f}")
    print(f"    Worst-case risk: {vals.max():.4f}")
    print(f"    Best-case risk:  {vals.min():.4f}")
    print(f"    Risk gap:       {vals.max() - vals.min():.4f}")
    print(f"    Std of risks:   {vals.std():.4f}")
    return vals.mean(), vals.max(), vals.max() - vals.min(), vals.std()


def main():
    df = load_data()

    # Prepare features and target
    X = df[FEATURE_COLS].values.astype(float)
    y = df['log_tmv'].values
    env_ids = df['env_id'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split: temporal (2018-2022 train, 2023-2025 test)
    years = df['year'].values
    train_mask = years <= 2022
    test_mask = years >= 2023

    X_train, y_train, env_train = X[train_mask], y[train_mask], env_ids[train_mask]
    X_test, y_test, env_test = X[test_mask], y[test_mask], env_ids[test_mask]

    # Only keep test environments that also appear in train
    common_envs = set(np.unique(env_train)) & set(np.unique(env_test))
    train_keep = np.isin(env_train, list(common_envs))
    test_keep = np.isin(env_test, list(common_envs))
    X_train, y_train, env_train = X_train[train_keep], y_train[train_keep], env_train[train_keep]
    X_test, y_test, env_test = X_test[test_keep], y_test[test_keep], env_test[test_keep]

    print(f"\nTrain: {len(y_train):,} obs, {len(np.unique(env_train))} envs (2018-2022)")
    print(f"Test:  {len(y_test):,} obs, {len(np.unique(env_test))} envs (2023-2025)")

    # ── 1. ERM ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. ERM (pooled Ridge)")
    print("=" * 60)
    erm_model, erm_train, erm_test = run_erm(X_train, y_train, X_test, y_test, env_train, env_test)
    print("\n  TRAIN:")
    erm_train_stats = summarize_risks(erm_train, "ERM")
    print("  TEST:")
    erm_test_stats = summarize_risks(erm_test, "ERM")

    # ── 2. IRM ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. IRM (λ=100)")
    print("=" * 60)
    irm_w, irm_train, irm_test = run_irm(
        X_train, y_train, env_train, X_test, y_test, env_test,
        lam=1.0, lr=0.001, n_iters=5000
    )
    print("\n  TRAIN:")
    irm_train_stats = summarize_risks(irm_train, "IRM")
    print("  TEST:")
    irm_test_stats = summarize_risks(irm_test, "IRM")

    # ── 3. V-REx ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. V-REx (β=100)")
    print("=" * 60)
    vrex_w, vrex_train, vrex_test = run_vrex(
        X_train, y_train, env_train, X_test, y_test, env_test,
        beta=1.0, lr=0.001, n_iters=5000
    )
    print("\n  TRAIN:")
    vrex_train_stats = summarize_risks(vrex_train, "V-REx")
    print("  TEST:")
    vrex_test_stats = summarize_risks(vrex_test, "V-REx")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY (TEST SET)")
    print("=" * 60)
    print(f"{'Method':<10} {'Mean Risk':>10} {'Worst Risk':>12} {'Risk Gap':>10} {'Std':>10}")
    print("-" * 55)
    for name, stats in [("ERM", erm_test_stats), ("IRM", irm_test_stats), ("V-REx", vrex_test_stats)]:
        mean_r, worst_r, gap, std = stats
        print(f"{name:<10} {mean_r:>10.4f} {worst_r:>12.4f} {gap:>10.4f} {std:>10.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, "irm_vrex_first_result.txt")
    with open(results_path, 'w') as f:
        f.write("IRM vs V-REx vs ERM — First Result\n")
        f.write("=" * 55 + "\n")
        f.write(f"Generated: 2026-03-09\n")
        f.write(f"Target: log(total_market_value)\n")
        f.write(f"Features: {FEATURE_COLS}\n")
        f.write(f"Train: 2018-2022 | Test: 2023-2025\n")
        f.write(f"Min env size: {MIN_ENV_SIZE}\n")
        f.write(f"Train obs: {len(y_train):,} | Test obs: {len(y_test):,}\n")
        f.write(f"Train envs: {len(np.unique(env_train))} | Test envs: {len(np.unique(env_test))}\n\n")
        f.write(f"{'Method':<10} {'Mean Risk':>10} {'Worst Risk':>12} {'Risk Gap':>10} {'Std':>10}\n")
        f.write("-" * 55 + "\n")
        for name, stats in [("ERM", erm_test_stats), ("IRM", irm_test_stats), ("V-REx", vrex_test_stats)]:
            mean_r, worst_r, gap, std = stats
            f.write(f"{name:<10} {mean_r:>10.4f} {worst_r:>12.4f} {gap:>10.4f} {std:>10.4f}\n")
    print(f"\nSaved results to {results_path}")


if __name__ == '__main__':
    main()
