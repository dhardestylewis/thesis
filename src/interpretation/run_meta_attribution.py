"""
run_meta_attribution.py
-----------------------
Formal registered meta-attribution experiment for Stage C.

Design goals (matching thesis critique):
  - Consensus features require EXPLICIT quantitative criteria, not heuristics.
  - Output three registered objects: consensus_features, model_specific_features,
    unstable_features, each with bootstrap uncertainty.
  - Until a cluster meets all consensus criteria it is called
    cross_model_consensus_features (not "invariant core").

Consensus rule (all conditions must hold):
  1. Top-quartile mean attribution share across model-period cells.
  2. Present (attribution_share > 0.05) in >= 75% of model-period cells.
  3. Bootstrap 95% CI for mean share excludes zero.
  4. Rank stability (Spearman rho across adjacent periods) >= 0.70.
"""

import json
import warnings

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR, save_registry

# ── Consensus thresholds (explicit; must match manuscript description) ──────
CONSENSUS_MIN_PRESENCE = 0.75      # fraction of model-period cells
CONSENSUS_MIN_RANK_STABILITY = 0.70 # Spearman rho across adjacent periods
CONSENSUS_PRESENCE_THRESHOLD = 0.05 # min share to count as "present" in a cell
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 42
ALPHA = 0.05  # CI level


def _bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP,
                  seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of values."""
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean()
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, 100 * ALPHA / 2)
    hi = np.percentile(boot_means, 100 * (1 - ALPHA / 2))
    return float(lo), float(hi)


def _rank_stability(df: pd.DataFrame,
                    cluster_col: str = "semantic_cluster",
                    period_col: str = "split_id",
                    value_col: str = "cluster_share") -> pd.Series:
    """
    Compute mean adjacent-period Spearman rank correlation per cluster.
    Returns a Series indexed by cluster name.
    """
    periods = sorted(df[period_col].unique())
    if len(periods) < 2:
        return pd.Series(dtype=float)

    corr_dfs = []
    for i in range(len(periods) - 1):
        left = (df[df[period_col] == periods[i]]
                .groupby(cluster_col)[value_col].mean())
        right = (df[df[period_col] == periods[i + 1]]
                 .groupby(cluster_col)[value_col].mean())
        combined = pd.concat([left, right], axis=1, keys=["a", "b"]).dropna()
        if len(combined) >= 3:
            corr = combined["a"].corr(combined["b"], method="spearman")
            corr_dfs.append(corr)

    if not corr_dfs:
        return pd.Series(dtype=float)

    # For simplicity, return overall mean rho as a scalar (same for all clusters)
    # A per-cluster stability would require leave-one-out; this is the global mean.
    mean_rho = float(np.mean(corr_dfs))
    # Return as Series over clusters so downstream merge works
    clusters = df[cluster_col].unique()
    return pd.Series(mean_rho, index=clusters, name="rank_stability")


def run_meta_attribution(
    consensus_min_presence: float = CONSENSUS_MIN_PRESENCE,
    consensus_min_rank_stability: float = CONSENSUS_MIN_RANK_STABILITY,
    n_boot: int = N_BOOTSTRAP,
) -> dict:
    """
    Build the formal meta-attribution object.

    Returns dict with keys:
        consensus_features        – meet ALL quantitative criteria
        model_specific_features   – high in one model family only
        unstable_features         – high variance across periods
        cluster_summary           – full per-cluster table (as records)
    """
    print("[+] Running Formal Meta-Attribution Experiment...")
    print(f"    Consensus rules: presence>={consensus_min_presence:.0%}, "
          f"rank_stability>={consensus_min_rank_stability:.2f}, "
          f"bootstrap CI excludes zero")

    # ── 1. Load interpretation registry ──────────────────────────────────────
    interp_path = REGISTRY_DIR / "interpretation_registry.parquet"
    if not interp_path.exists():
        print("    [!] Error: No interpretation data found. Run training first.")
        return {}

    df_attr = pd.read_parquet(interp_path)

    # ── 2. Load semantic mapping ──────────────────────────────────────────────
    mapping_path = ROOT_DIR / "configs" / "features" / "semantic_clusters.yaml"
    with open(mapping_path, "r") as f:
        config = yaml.safe_load(f)

    clusters = config["clusters"]
    feat_to_cluster = {
        feat: cluster_name
        for cluster_name, features in clusters.items()
        for feat in features
    }

    df_attr["semantic_cluster"] = (
        df_attr["feature_name"].map(feat_to_cluster).fillna("Other")
    )

    # ── 3. Compute cluster shares per (model_family, split_id) cell ──────────
    # Raw aggregation to cluster level within each model × period cell
    cell_agg = (
        df_attr
        .groupby(["model_family", "split_id", "semantic_cluster"])["attribution_value"]
        .sum()
        .reset_index()
    )

    # Normalize within each cell to get share
    cell_totals = (
        cell_agg
        .groupby(["model_family", "split_id"])["attribution_value"]
        .sum()
        .rename("cell_total")
    )
    cell_agg = cell_agg.merge(cell_totals, on=["model_family", "split_id"])
    cell_agg["cluster_share"] = cell_agg["attribution_value"] / cell_agg["cell_total"]

    # Save clustered registry
    save_registry(cell_agg, "interpretation_registry_clustered")

    # ── 4. Compute per-cluster statistics ────────────────────────────────────
    n_cells = cell_agg.groupby("semantic_cluster").size().rename("n_cells")
    n_total_cells = cell_agg[["model_family", "split_id"]].drop_duplicates().shape[0]

    summary_rows = []
    for cluster, grp in cell_agg.groupby("semantic_cluster"):
        shares = grp["cluster_share"].values
        mean_share = float(np.mean(shares))
        std_share = float(np.std(shares))

        ci_lo, ci_hi = _bootstrap_ci(shares, n_boot=n_boot)
        ci_excludes_zero = ci_lo > 0.0

        presence = float((shares >= CONSENSUS_PRESENCE_THRESHOLD).mean())

        summary_rows.append({
            "semantic_cluster": cluster,
            "mean_share": mean_share,
            "std_share": std_share,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_excludes_zero": ci_excludes_zero,
            "presence_fraction": presence,
            "n_cells": len(shares),
        })

    summary = pd.DataFrame(summary_rows).sort_values("mean_share", ascending=False)

    # ── 5. Rank stability (global Spearman rho across adjacent periods) ───────
    rank_stab = _rank_stability(cell_agg)
    summary["rank_stability"] = summary["semantic_cluster"].map(rank_stab)
    # Fill NaN if not computable
    summary["rank_stability"] = summary["rank_stability"].fillna(0.0)

    # ── 6. Top-quartile threshold ─────────────────────────────────────────────
    q75_share = summary["mean_share"].quantile(0.75)

    # ── 7. Apply formal consensus criteria ───────────────────────────────────
    # A cluster is a "consensus feature" only if ALL four conditions hold.
    summary["meets_top_quartile"] = summary["mean_share"] >= q75_share
    summary["meets_presence"] = summary["presence_fraction"] >= consensus_min_presence
    summary["meets_ci"] = summary["ci_excludes_zero"]
    summary["meets_rank_stability"] = summary["rank_stability"] >= consensus_min_rank_stability

    summary["is_consensus"] = (
        summary["meets_top_quartile"]
        & summary["meets_presence"]
        & summary["meets_ci"]
        & summary["meets_rank_stability"]
    )

    # ── 8. Model-specific: top quartile in >=1 model but NOT consensus ────────
    model_top = (
        cell_agg
        .groupby(["model_family", "semantic_cluster"])["cluster_share"]
        .mean()
        .reset_index()
    )
    model_top_q75 = model_top.groupby("model_family")["cluster_share"].transform(
        lambda x: x >= x.quantile(0.75)
    )
    model_specific_clusters = set(
        model_top.loc[model_top_q75, "semantic_cluster"].unique()
    )

    summary["is_model_specific"] = (
        summary["semantic_cluster"].isin(model_specific_clusters)
        & ~summary["is_consensus"]
    )

    # ── 9. Unstable: high std relative to mean (CV > 0.5), or low rank stability
    summary["coeff_variation"] = summary["std_share"] / (summary["mean_share"] + 1e-9)
    summary["is_unstable"] = (
        (summary["coeff_variation"] > 0.5)
        | (summary["rank_stability"] < 0.50)
    )

    # ── 10. Export feature lists ───────────────────────────────────────────────
    consensus_features = summary.loc[summary["is_consensus"], "semantic_cluster"].tolist()
    model_specific_features = summary.loc[
        summary["is_model_specific"], "semantic_cluster"
    ].tolist()
    unstable_features = summary.loc[summary["is_unstable"], "semantic_cluster"].tolist()

    # ── 11. Print summary ─────────────────────────────────────────────────────
    print(f"\n>>> META-ATTRIBUTION RESULTS (n_bootstrap={n_boot}) <<<")
    print(f"    Total model-period cells: {n_total_cells}")
    print(f"    Top-quartile share threshold: {q75_share:.3f}")
    print()
    print(summary[[
        "semantic_cluster", "mean_share", "presence_fraction",
        "ci_lo", "ci_hi", "rank_stability",
        "is_consensus", "is_model_specific", "is_unstable"
    ]].to_string(index=False))

    print(f"\n[*] cross_model_consensus_features (all 4 criteria met): {consensus_features}")
    print(f"[*] model_specific_features: {model_specific_features}")
    print(f"[*] unstable_features: {unstable_features}")

    if not consensus_features:
        warnings.warn(
            "No features met all consensus criteria. "
            "Do not use the term 'invariant core' in manuscript prose until "
            "at least one cluster meets all four quantitative thresholds.",
            stacklevel=2,
        )

    # ── 12. Save formal output object ─────────────────────────────────────────
    output_obj = {
        "meta_attribution_version": "v2",
        "consensus_criteria": {
            "min_presence_fraction": consensus_min_presence,
            "min_rank_stability": consensus_min_rank_stability,
            "bootstrap_ci_excludes_zero": True,
            "must_be_top_quartile": True,
            "presence_threshold_per_cell": CONSENSUS_PRESENCE_THRESHOLD,
            "n_bootstrap": n_boot,
            "alpha": ALPHA,
        },
        "q75_share_threshold": q75_share,
        "n_total_cells": n_total_cells,
        "cross_model_consensus_features": consensus_features,
        "model_specific_features": model_specific_features,
        "unstable_features": unstable_features,
        "cluster_summary": summary.to_dict(orient="records"),
    }

    out_path = REGISTRY_DIR / "meta_attribution_object.json"
    with open(out_path, "w") as f:
        json.dump(output_obj, f, indent=4, default=str)

    print(f"\n[+] Saved meta-attribution object -> {out_path}")
    return output_obj


if __name__ == "__main__":
    run_meta_attribution()
