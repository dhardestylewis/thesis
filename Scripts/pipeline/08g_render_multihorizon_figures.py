"""
08g_render_multihorizon_figures.py

Reads the multi-model multi-horizon OOT CSVs (biweekly + annualized) and
renders the full figure suite matching the existing thesis temporal drift
exhibit style:
  1. PR-AUC heatmap by (Horizon x Test_Year), faceted by model
  2. Lift decay curve (PR-AUC / naive base rate) by horizon, faceted by model family
  3. Architecture dominance bump chart (which model wins each cell)
  4. ROC-AUC by horizon barplot comparison
  5. Annualized: same 4 figures for the 1/2/3-year panel

Output directory: Thesis_Draft/Draft_v1/Figures/exhibits/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS  = ROOT / "artifacts"
FIGURES    = ROOT / "Thesis_Draft/Draft_v1/Figures/exhibits"
FIGURES.mkdir(parents=True, exist_ok=True)

BW_CSV  = ARTIFACTS / "multihorizon_multicutoff_all_models.csv"
ANN_CSV = ARTIFACTS / "annualized_multihorizon_multicutoff_all_models.csv"

# ── Style ──────────────────────────────────────────────────────────────────
FAMILY_COLORS = {
    "Tree":   "#2563EB",   # blue
    "Linear": "#DC2626",   # red
    "Deep":   "#16A34A",   # green
}
MODEL_MARKERS = {
    "CatBoost":    "o",
    "RandomForest":"s",
    "LogisticL2":  "^",
    "LogisticL1":  "D",
    "MLP":         "P",
}
HORIZON_ORDER_BW  = ["14_Days", "3_Months", "6_Months", "1_Year", "2_Years"]
HORIZON_ORDER_ANN = ["1_Year", "2_Years", "3_Years"]

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load(path, label):
    if not path.exists():
        print(f"[SKIP] {label} CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    df["Lift"] = df["PR_AUC"] / df["Naive_PR_AUC"].clip(lower=1e-6)
    return df


# ── Fig 1: PR-AUC heatmap per model ────────────────────────────────────────
def fig_prauc_heatmap(df, horizon_order, tag, title_prefix):
    models = df["Model"].unique()
    test_years = sorted(df["Test_Year"].unique())
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizon_order, ordered=True)

    ncols = min(3, len(models))
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()

    for ax, model in zip(axes, models):
        sub = df[df["Model"] == model].pivot(index="Horizon", columns="Test_Year", values="PR_AUC")
        sub = sub.reindex(horizon_order)
        sns.heatmap(sub, ax=ax, cmap="Blues", annot=True, fmt=".3f",
                    vmin=0, vmax=sub.values.max() * 1.1,
                    linewidths=0.4, cbar_kws={"shrink": 0.7})
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Year")
        ax.set_ylabel("Horizon")

    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.suptitle(f"{title_prefix} — PR-AUC by Horizon × Test Year", fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGURES / f"fig_multihorizon_prauc_heatmap_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] {out.name}")


# ── Fig 2: Lift decay curve by horizon ─────────────────────────────────────
def fig_lift_decay(df, horizon_order, tag, title_prefix):
    df = df.copy()
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizon_order, ordered=True)
    mean_lift = df.groupby(["Horizon", "Model", "Model_Family"])["Lift"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for (model, family), grp in mean_lift.groupby(["Model", "Model_Family"]):
        grp = grp.sort_values("Horizon")
        color = FAMILY_COLORS.get(family, "#555")
        marker = MODEL_MARKERS.get(model, "o")
        ax.plot(grp["Horizon"].astype(str), grp["Lift"],
                marker=marker, color=color, label=model, linewidth=1.8)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Naive baseline (lift=1)")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("PR-AUC Lift over Naive")
    ax.set_title(f"{title_prefix} — PR-AUC Lift Decay by Horizon", fontsize=12)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = FIGURES / f"fig_multihorizon_lift_decay_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] {out.name}")


# ── Fig 3: Architecture dominance bump chart ────────────────────────────────
def fig_dominance_bump(df, horizon_order, tag, title_prefix):
    df = df.copy()
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizon_order, ordered=True)
    winners = (df.groupby(["Horizon", "Test_Year"])
                 .apply(lambda g: g.loc[g["PR_AUC"].idxmax(), "Model"])
                 .reset_index(name="Winner"))
    winners["Horizon_idx"] = winners["Horizon"].cat.codes
    winners["x"] = winners["Horizon_idx"].astype(str) + "_" + winners["Test_Year"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 4))
    for _, row in winners.iterrows():
        family = df.loc[df["Model"] == row["Winner"], "Model_Family"].iloc[0]
        color = FAMILY_COLORS.get(family, "#555")
        ax.scatter(f"{row['Horizon']} / {row['Test_Year']}", row["Winner"],
                   color=color, s=120, zorder=3)

    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=patches, title="Model Family", fontsize=8)
    ax.set_xlabel("Horizon / Test Year")
    ax.set_ylabel("Winning Model")
    ax.set_title(f"{title_prefix} — Architecture Dominance by Cell", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    out = FIGURES / f"fig_multihorizon_dominance_bump_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] {out.name}")


# ── Fig 4: Mean ROC-AUC by horizon barplot ─────────────────────────────────
def fig_roc_barplot(df, horizon_order, tag, title_prefix):
    df = df.copy()
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizon_order, ordered=True)
    mean_roc = df.groupby(["Horizon", "Model", "Model_Family"])["ROC_AUC"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(horizon_order))
    models = mean_roc["Model"].unique()
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        sub = mean_roc[mean_roc["Model"] == model].set_index("Horizon").reindex(horizon_order)
        family = df.loc[df["Model"] == model, "Model_Family"].iloc[0]
        color = FAMILY_COLORS.get(family, "#555")
        ax.bar(x + i * width - 0.4 + width/2, sub["ROC_AUC"].fillna(0),
               width=width * 0.9, color=color, alpha=0.7, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(horizon_order, rotation=30, ha="right")
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_title(f"{title_prefix} — Mean ROC-AUC by Horizon and Model", fontsize=12)
    ax.legend(ncol=2, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    out = FIGURES / f"fig_multihorizon_roc_barplot_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] {out.name}")


def run():
    print("\n=== Rendering Multi-Horizon Figure Suite ===\n")

    bw = load(BW_CSV, "Biweekly")
    if bw is not None:
        print("--- Biweekly Panel ---")
        fig_prauc_heatmap(bw,  HORIZON_ORDER_BW,  "biweekly", "Biweekly Walk-Forward")
        fig_lift_decay(bw,     HORIZON_ORDER_BW,  "biweekly", "Biweekly Walk-Forward")
        fig_dominance_bump(bw, HORIZON_ORDER_BW,  "biweekly", "Biweekly Walk-Forward")
        fig_roc_barplot(bw,    HORIZON_ORDER_BW,  "biweekly", "Biweekly Walk-Forward")

    ann = load(ANN_CSV, "Annualized")
    if ann is not None:
        print("--- Annualized Panel ---")
        fig_prauc_heatmap(ann,  HORIZON_ORDER_ANN, "annualized", "Annualized Walk-Forward")
        fig_lift_decay(ann,     HORIZON_ORDER_ANN, "annualized", "Annualized Walk-Forward")
        fig_dominance_bump(ann, HORIZON_ORDER_ANN, "annualized", "Annualized Walk-Forward")
        fig_roc_barplot(ann,    HORIZON_ORDER_ANN, "annualized", "Annualized Walk-Forward")

    print(f"\n[+] All figures saved to {FIGURES}")


if __name__ == "__main__":
    run()
