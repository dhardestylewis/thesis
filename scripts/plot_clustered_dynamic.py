"""
plot_clustered_dynamic.py
Faceted horizontal bar chart: Top-4 semantic feature clusters per model per anchor year.
Visually contrasts the conceptual drivers (Demographic vs. Spatial/Built Environment)
between CatBoost and LightGBM across temporal anchors.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = "c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track1_Predictive/Metrics/clustered_stability_H0.csv"
OUT_PATH = "c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track1_Predictive/Figures/fig_clustered_dynamic.pdf"
PNG_PATH = OUT_PATH.replace(".pdf", ".png")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    print(f"[!] Error: {CSV_PATH} not found.")
    exit(1)

df = pd.read_csv(CSV_PATH)
df = df.sort_values(["Model", "Anchor", "Share_Pct"], ascending=[True, True, False])

TOP_N = 4
ANCHORS = sorted(df["Anchor"].unique())
MODELS = ["CatBoost", "LightGBM"]

# Categorization for color coding
SPATIAL_GROUPS = {
    "Filing Timeline", "Structure Age", "Parcel Scale", "Zoning Density",
    "Improvement Scale", "Property Valuation", "Land Use Classification",
    "Historical Protest Activity", "Latitude", "Longitude", "Nearby Geoid",
    "Median Neighbor Far Cluster", "Max Far Cluster", "Min Lot Sqft",
    "Median Structure Age", "Floor Area Ratio", "Unit Count", "Site Area",
    "Account Number Formatted", "Appraisal District Id", "Tax Year", "Taxing Unit Id",
    "Median Sqft Cluster", "Std Appraised Value Cluster", "Improvement Ratio",
    "Constrained Area", "Zoning Case Geoid", "Parcel Identity Cluster",
    "Improvement Market Value", "Appraised Value", "Assessed Value", "Taxable Value"
}

DEMO_GROUPS = {
    "Housing Tenure", "Demographic Composition", "Neighborhood Income & Rent",
    "Median Household Income", "Renter Share Cluster", "Total Population Cluster",
    "Median Age", "Race & Ethnicity", "Occupancy Cluster", "Home Value Cluster",
    "Property Owner Identity", "Neighbor Comm Share", "Neighbor Sf Share", "Neighbor Mf Share",
    "Bisg White 200Ft", "Bisg Black 200Ft", "Bisg Asian 200Ft", "Bisg Hispanic 200Ft",
    "Bisg White Nbr", "Bisg Black Nbr", "Bisg Asian Nbr", "Bisg Hispanic Nbr"
}

# ── Color palette ─────────────────────────────────────────────────────────────
DEMO_COLOR     = "#4C72B0"   # cool blue  → demographics/socio-economic
SPATIAL_COLOR  = "#DD8452"   # warm amber → spatiotemporal/structural/built environment
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

def get_color(group_name):
    if group_name in SPATIAL_GROUPS:
        return SPATIAL_COLOR
    if group_name in DEMO_GROUPS:
        return DEMO_COLOR
    # Fallback/Default
    return "#8C8C8C" # Grey for 'Other' or unknown

for row_i, anchor in enumerate(ANCHORS):
    for col_i, model in enumerate(MODELS):
        ax = axes[row_i][col_i]
        ax.set_facecolor(BG_COLOR)

        sub = df[(df["Anchor"] == anchor) & (df["Model"] == model)].head(TOP_N)
        # Reverse so highest share is at top
        sub = sub.iloc[::-1].reset_index(drop=True)

        groups  = sub["Group"].tolist()
        shares  = sub["Share_Pct"].tolist()
        colors  = [get_color(g) for g in groups]

        y_pos = np.arange(len(groups))

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
        ax.set_yticklabels(groups, fontsize=7.5, color=LABEL_COLOR)
        ax.set_xlim(0, max(shares + [30]) * 1.25) # Ensure some min width
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
demo_patch    = mpatches.Patch(color=DEMO_COLOR,    label="Socio-Economic / Demographic Clusters")
spatial_patch = mpatches.Patch(color=SPATIAL_COLOR, label="Built Environment / Spatial / Time Clusters")

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
    "Top-4 Semantic Feature Clusters by Anchor Year and Model (Dynamic View)",
    fontsize=13, fontweight="bold", color=LABEL_COLOR, y=1.01,
)

fig.text(
    0.5, -0.04,
    "Note: Orange bars indicate clusters related to the built environment, zoning, or timeline; blue bars indicate socio-economic or neighborhood demographic clusters.\n"
    "This view aggregates feature importance into conceptual themes to assess structural stability across models.",
    ha="center", fontsize=8, color="#555555", style="italic",
    wrap=True,
)

plt.tight_layout(h_pad=1.8, w_pad=3.2) # Increased w_pad for longer cluster names

plt.savefig(OUT_PATH, format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(PNG_PATH, format="png", bbox_inches="tight", dpi=180)
plt.close()

print(f"[OK] PDF saved: {OUT_PATH}")
print(f"[OK] PNG saved: {PNG_PATH}")
