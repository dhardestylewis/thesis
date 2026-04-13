"""
plot_unclustered_dynamic.py
Faceted horizontal bar chart: Top-4 unclustered features per model per anchor year.
Spatiotemporal/structural features are highlighted in a distinct color to visually
contrast LightGBM's reliance on them versus CatBoost's demographic focus.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = "c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track1_Predictive/Metrics/unclustered_stability_H0.csv"
OUT_PATH = "c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track1_Predictive/Figures/fig_unclustered_dynamic.pdf"
PNG_PATH = OUT_PATH.replace(".pdf", ".png")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.sort_values(["Model", "Anchor", "Share_Pct"], ascending=[True, True, False])

TOP_N = 4
ANCHORS = sorted(df["Anchor"].unique())
MODELS = ["CatBoost", "LightGBM"]

# Features considered spatiotemporal / structural size (highlighted in plot)
SPATIOTEMPORAL = {
    "Year", "Site Area", "Latitude", "Longitude",
    "Median Structure Age", "Structure Age", "Median Sqft", "Sqft",
    "Std Appraised Value",
}

# ── Color palette ─────────────────────────────────────────────────────────────
# Muted slate-blue for demographic features, orange-amber for spatiotemporal
DEMO_COLOR     = "#4C72B0"   # cool blue  → demographics
SPATIAL_COLOR  = "#DD8452"   # warm amber → spatiotemporal/structural
BG_COLOR       = "#F7F7F7"
GRID_COLOR     = "#DCDCDC"
LABEL_COLOR    = "#2C2C2C"

# ── Typography ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Layout ────────────────────────────────────────────────────────────────────
N_ROWS = len(ANCHORS)   # 5
N_COLS = len(MODELS)    # 2

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(12, 14),
    sharex=False,
)
fig.patch.set_facecolor(BG_COLOR)

for row_i, anchor in enumerate(ANCHORS):
    for col_i, model in enumerate(MODELS):
        ax = axes[row_i][col_i]
        ax.set_facecolor(BG_COLOR)

        sub = df[(df["Anchor"] == anchor) & (df["Model"] == model)].head(TOP_N)
        # Reverse so highest share is at top
        sub = sub.iloc[::-1].reset_index(drop=True)

        features  = sub["Feature"].tolist()
        shares    = sub["Share_Pct"].tolist()
        colors    = [SPATIAL_COLOR if f in SPATIOTEMPORAL else DEMO_COLOR for f in features]

        y_pos = np.arange(len(features))

        bars = ax.barh(y_pos, shares, color=colors, height=0.55,
                       edgecolor="white", linewidth=0.6)

        # Value labels inside bars
        for bar, val in zip(bars, shares):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left",
                fontsize=6.8, color=LABEL_COLOR, fontweight="bold"
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=7.5, color=LABEL_COLOR)
        ax.set_xlim(0, max(shares) * 1.35)
        ax.axvline(0, color=LABEL_COLOR, linewidth=0.5)
        ax.set_xlabel("Attribution Share (%)", fontsize=7.5)

        # Grid on x only
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)

        # Column header (model name) only on top row
        if row_i == 0:
            ax.set_title(model, fontsize=11, fontweight="bold",
                         color=LABEL_COLOR, pad=8)

        # Row label (anchor year) only on leftmost column
        if col_i == 0:
            ax.set_ylabel(f"{anchor}", fontsize=10, fontweight="bold",
                          color=LABEL_COLOR, rotation=0, labelpad=36, va="center")

# ── Legend ────────────────────────────────────────────────────────────────────
demo_patch    = mpatches.Patch(color=DEMO_COLOR,    label="Socio-Economic / Demographic")
spatial_patch = mpatches.Patch(color=SPATIAL_COLOR, label="Spatiotemporal / Structural Size")

fig.legend(
    handles=[demo_patch, spatial_patch],
    loc="lower center",
    ncol=2,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)

# ── Title and caption ─────────────────────────────────────────────────────────
fig.suptitle(
    "Top-4 Unclustered Features by Anchor Year and Model (Dynamic View)",
    fontsize=13, fontweight="bold", color=LABEL_COLOR, y=1.01,
)

fig.text(
    0.5, -0.04,
    "Note: Orange bars indicate spatiotemporal/structural size features; blue bars indicate socio-economic or neighborhood demographic features.\n"
    "LightGBM consistently draws on temporal and spatial splits, while CatBoost concentrates attribution in demographics.",
    ha="center", fontsize=8, color="#555555", style="italic",
    wrap=True,
)

plt.tight_layout(h_pad=1.8, w_pad=2.5)

plt.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(PNG_PATH, format="png", bbox_inches="tight", dpi=180)
plt.close()

print(f"[OK] PDF saved: {OUT_PATH}")
print(f"[OK] PNG saved: {PNG_PATH}")
