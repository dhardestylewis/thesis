"""
icp_feature_identification.py — Invariant Causal Prediction (Peters et al. 2016)
=================================================================================
Implements Method II from Peters, Bühlmann, Meinshausen (2016) JRSS-B:
"Causal inference by using invariant prediction: identification and confidence intervals"

For each candidate feature subset S ⊆ {1,...,p}:
  1. Fit a pooled linear regression Y ~ X_S on all data.
  2. Compute residuals R = Y - X_S @ β_hat.
  3. For each environment e ∈ E:
     a. Two-sample t-test: H0: mean(R_e) == mean(R_{-e})
     b. F-test (Levene's): H0: var(R_e) == var(R_{-e})
  4. Bonferroni-correct across |E| environments.
  5. Combine the two p-values: p_S = 2 * min(p_mean, p_var).
  6. Accept S if p_S >= α.

The identified causal predictors are:
  Ŝ* = ⋂_{S: not rejected} S

This gives us the NAMED features (not just a latent representation) that are
causally invariant across all municipal zoning interventions, with formal
confidence guarantees: P[Ŝ* ⊆ S*] >= 1 - α.

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings
import os
import time

warnings.filterwarnings('ignore')

# Configuration
ALPHA = 0.05          # Significance level
MAX_SUBSET_SIZE = 3   # Limit combinatorial explosion (test up to 3-feature subsets)
MIN_ENV_SIZE = 10     # Minimum observations per environment for valid testing
SAMPLED_SIZE = 35000

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

# Feature names (the raw numeric predictors we can name)
NUMERIC_FEATURES = [
    'total_market_value',
    'deed_acreage',
    'improvement_sq_ft'
]

def load_data():
    """Load panel data and merge environment labels."""
    print("Loading panel data...")
    cols = NUMERIC_FEATURES + ['year', 'protest', 'standardized_tcad_id']
    panel = pd.read_csv(PANEL_PATH, usecols=cols, low_memory=False)
    panel['improvement_sq_ft'] = pd.to_numeric(panel['improvement_sq_ft'], errors='coerce')
    panel = panel[panel['year'] <= 2024]
    
    print("Loading environment assignments...")
    env = pd.read_csv(ENV_PATH).rename(columns={'CASE_NUMBER': 'env_id'})
    
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BACKGROUND')
    
    # Filter environments by minimum size
    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]
    
    # Sample for tractability
    positives = df[df['protest'] == 1]
    negatives = df[df['protest'] == 0].sample(n=SAMPLED_SIZE - len(positives), random_state=42)
    df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clean numeric features
    for col in NUMERIC_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df[NUMERIC_FEATURES])
    y = df['protest'].values.astype(np.float64)
    envs = df['env_id'].values
    
    unique_envs = [e for e in df['env_id'].unique() if e != 'BACKGROUND']
    
    print(f"Dataset: {len(df):,} rows | {len(NUMERIC_FEATURES)} numeric features | {len(unique_envs)} environments")
    return X, y, envs, unique_envs


def test_subset_invariance(X, y, envs, unique_envs, feature_indices, feature_names):
    """
    Peters et al. Method II (Logistic Extension): Test whether the residuals 
    of logistic regression on feature subset S have identical mean and variance 
    across all environments.
    
    For classification: R = Y - f_hat(X), where f_hat is the predicted probability.
    
    Returns: (combined_p, raw_min_p_mean, raw_min_p_var, accepted)
    """
    from sklearn.linear_model import LogisticRegression
    
    X_S = X[:, feature_indices]
    
    if X_S.shape[1] == 0:
        # Empty set: predict with intercept only (base rate)
        f_hat = np.full(len(y), y.mean())
    else:
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_S, y)
        f_hat = clf.predict_proba(X_S)[:, 1]
    
    residuals = y - f_hat
    
    # For each environment e, test mean and variance of residuals
    p_means = []
    p_vars = []
    
    for e in unique_envs:
        mask_e = envs == e
        n_e = mask_e.sum()
        if n_e < 5:
            continue
            
        R_e = residuals[mask_e]
        R_not_e = residuals[~mask_e]
        
        # Two-sample t-test for equality of means
        t_stat, p_mean = stats.ttest_ind(R_e, R_not_e, equal_var=False)
        p_means.append(p_mean)
        
        # Levene's test for equality of variances
        lev_stat, p_var = stats.levene(R_e, R_not_e)
        p_vars.append(p_var)
    
    if not p_means:
        return 0.0, 1.0, 1.0, False
    
    # Raw minimum p-values (before Bonferroni)
    raw_min_p_mean = min(p_means)
    raw_min_p_var = min(p_vars)
    
    # Bonferroni correction across environments
    min_p_mean = raw_min_p_mean * len(p_means)
    min_p_var = raw_min_p_var * len(p_vars)
    
    # Combine: p_S = 2 * min(p_mean_bonf, p_var_bonf)
    combined_p = 2 * min(min_p_mean, min_p_var)
    combined_p = min(combined_p, 1.0)
    
    accepted = combined_p >= ALPHA
    return combined_p, raw_min_p_mean, raw_min_p_var, accepted


def run_icp(X, y, envs, unique_envs):
    """
    Run the full ICP procedure: test all subsets up to MAX_SUBSET_SIZE.
    
    Ŝ* = ⋂_{S: not rejected} S
    """
    p = X.shape[1]
    feature_names = NUMERIC_FEATURES
    
    accepted_subsets = []
    rejected_subsets = []
    
    all_subsets = []
    
    # Generate all subsets from size 0 to MAX_SUBSET_SIZE
    for size in range(0, min(MAX_SUBSET_SIZE + 1, p + 1)):
        for combo in combinations(range(p), size):
            all_subsets.append(combo)
    
    print(f"\nTesting {len(all_subsets)} feature subsets for invariance (α = {ALPHA})...")
    print("=" * 80)
    
    for subset_indices in all_subsets:
        subset_names = [feature_names[i] for i in subset_indices]
        label = "{" + ", ".join(subset_names) + "}" if subset_names else "∅ (empty set)"
        
        p_value, raw_p_mean, raw_p_var, accepted = test_subset_invariance(
            X, y, envs, unique_envs, list(subset_indices), subset_names
        )
        
        status = "ACCEPTED ✓" if accepted else "REJECTED ✗"
        print(f"  S = {label:<55s} p_bonf = {p_value:.6f}  raw_mean = {raw_p_mean:.2e}  raw_var = {raw_p_var:.2e}  {status}")
        
        if accepted:
            accepted_subsets.append(set(subset_indices))
        else:
            rejected_subsets.append(set(subset_indices))
    
    # Compute Ŝ* = intersection of all accepted subsets
    print("\n" + "=" * 80)
    print("ICP RESULTS: Identified Causal Predictors")
    print("=" * 80)
    
    if not accepted_subsets:
        print("NO subsets were accepted. The ICP procedure found no invariant set.")
        print("This means the data may violate the linear causal model assumption,")
        print("or the environments are too diverse for any linear subset to be invariant.")
        return set()
    
    # The identified causal predictors
    S_hat = accepted_subsets[0]
    for s in accepted_subsets[1:]:
        S_hat = S_hat.intersection(s)
    
    if not S_hat:
        print("Ŝ* = ∅ (empty set)")
        print("Multiple accepted subsets exist but they share NO common features.")
        print("This means every feature appears in at least one rejected configuration,")
        print("so we cannot make a confident causal claim about any individual feature.")
        print("\nAccepted subsets were:")
        for s in accepted_subsets:
            names = [feature_names[i] for i in s] if s else ["(empty)"]
            print(f"  {{{', '.join(names)}}}")
    else:
        causal_features = [feature_names[i] for i in sorted(S_hat)]
        print(f"Ŝ* = {{{', '.join(causal_features)}}}")
        print(f"\nWith confidence ≥ {1 - ALPHA:.0%}, the following features are")
        print(f"CAUSALLY INVARIANT predictors of protest across all zoning interventions:")
        for f in causal_features:
            print(f"  → {f}")
    
    # Also compute confidence intervals for the causal coefficients
    if S_hat:
        S_indices = sorted(list(S_hat))
        X_S = X[:, S_indices]
        XtX = X_S.T @ X_S
        beta = np.linalg.solve(XtX, X_S.T @ y)
        y_hat = X_S @ beta
        residuals = y - y_hat
        n = len(y)
        k = len(S_indices)
        sigma2 = np.sum(residuals**2) / (n - k)
        cov_beta = sigma2 * np.linalg.inv(XtX)
        se = np.sqrt(np.diag(cov_beta))
        
        t_crit = stats.t.ppf(1 - ALPHA / (2 * k), df=n - k)  # Bonferroni-corrected
        
        print(f"\nCausal Coefficients (Bonferroni-corrected {1 - ALPHA:.0%} CI):")
        print(f"{'Feature':<25s} {'β̂':>10s} {'SE':>10s} {'CI Lower':>10s} {'CI Upper':>10s}")
        print("-" * 70)
        for j, idx in enumerate(S_indices):
            ci_lo = beta[j] - t_crit * se[j]
            ci_hi = beta[j] + t_crit * se[j]
            print(f"{feature_names[idx]:<25s} {beta[j]:>10.6f} {se[j]:>10.6f} {ci_lo:>10.6f} {ci_hi:>10.6f}")
    
    return S_hat


def main():
    t0 = time.time()
    X, y, envs, unique_envs = load_data()
    S_hat = run_icp(X, y, envs, unique_envs)
    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
