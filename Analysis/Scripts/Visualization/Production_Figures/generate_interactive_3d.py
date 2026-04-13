import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Re-open stdout in UTF-8 so print() works on Windows cp1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
generate_interactive_3d.py
--------------------------
1.  Reads rolling_origin_drift.json
2.  Produces docs/index.html  — a standalone, self-contained Plotly 3-D
    surface that GitHub Pages will host at:
        https://dhardestylewis.github.io/thesis/
3.  Produces qr_temporal_drift.png  — a small QR code that links to the
    same URL, intended for embedding in the bottom-right corner of the
    static 3-panel PDF figure.
4.  Rebuilds the static 3-panel figure with the QR code stamp.
"""

import os, json, sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import qrcode
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = r"C:\Users\dhl\data\thesis\thesis"
DATA_IN    = os.path.join(ROOT_DIR, "Analysis", "Output",
                          "Track1_Predictive", "rolling_origin_drift.json")
DOCS_DIR   = os.path.join(ROOT_DIR, "docs")
FIG_DIR    = os.path.join(ROOT_DIR, "Thesis_Draft", "Draft_v1",
                          "Figures", "Chapter4")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GITHUB_PAGES_URL = "https://dhardestylewis.github.io/thesis/"
HTML_OUT   = os.path.join(DOCS_DIR,  "index.html")
QR_OUT     = os.path.join(SCRIPT_DIR, "qr_temporal_drift.png")
STATIC_OUT = os.path.join(FIG_DIR,   "Fig_3D_Temporal_Drift.png")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
def load_df():
    if not os.path.exists(DATA_IN):
        print(f"[!] Data file not found: {DATA_IN}")
        return None
    with open(DATA_IN) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["Anchor_Year"] = df["Anchor"].str.replace("Pre-", "").astype(int)
    return df

# ── 1. Interactive Plotly HTML ────────────────────────────────────────────────
def build_html(df):
    df_cat = df[df["Model"] == "CatBoost"].dropna(subset=["PR-AUC"])
    df_tab = df[df["Model"] == "TabNet(LS+Pruning)"].dropna(subset=["PR-AUC"])

    traces = []

    if not df_cat.empty:
        traces.append(go.Mesh3d(
            x=df_cat["Anchor_Year"],
            y=df_cat["Evaluate_Year"],
            z=df_cat["PR-AUC"],
            colorscale="Blues",
            intensity=df_cat["PR-AUC"],
            opacity=0.82,
            name="CatBoost — Structural Guardrail",
            showlegend=True,
            hovertemplate=(
                "<b>CatBoost</b><br>"
                "Anchor: %{x}<br>Eval Year: %{y}<br>PR-AUC: %{z:.3f}<extra></extra>"
            ),
        ))

    if not df_tab.empty:
        traces.append(go.Mesh3d(
            x=df_tab["Anchor_Year"],
            y=df_tab["Evaluate_Year"],
            z=df_tab["PR-AUC"],
            colorscale="Reds",
            intensity=df_tab["PR-AUC"],
            opacity=0.62,
            name="TabNet (LS+Pruning) — Deep Interpolation",
            showlegend=True,
            hovertemplate=(
                "<b>TabNet Pruned</b><br>"
                "Anchor: %{x}<br>Eval Year: %{y}<br>PR-AUC: %{z:.3f}<extra></extra>"
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                "3D Temporal Drift Topology<br>"
                "<sup>Drag to rotate · Scroll to zoom · Hover for values</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=18, family="Georgia, serif"),
        ),
        scene=dict(
            xaxis_title="Anchor Year (Training Bound)",
            yaxis_title="Horizon Year (Eval Target)",
            zaxis_title="PR-AUC Performance",
            xaxis=dict(backgroundcolor="#f7f7f7"),
            yaxis=dict(backgroundcolor="#f0f0f0"),
            zaxis=dict(backgroundcolor="#e8e8e8"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        legend=dict(
            x=0.02, y=0.95,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=80, b=0),
        annotations=[dict(
            text=(
                "Figure 6 · Predicting Formal Protest Petitions Against Housing "
                "Development: Evidence from Austin, Texas<br>"
                "Daniel Hardesty Lewis · Columbia GSAPP · 2026"
            ),
            showarrow=False, xref="paper", yref="paper",
            x=0.5, y=-0.04, xanchor="center",
            font=dict(size=11, color="#666666"),
        )],
    )

    # Full self-contained HTML (no CDN dependency)
    html_str = fig.to_html(
        full_html=True,
        include_plotlyjs=True,   # embeds ~3 MB of JS — no CDN needed offline
        config={"displayModeBar": True, "scrollZoom": True},
    )
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"[+] Interactive HTML -> {HTML_OUT}")

# ── 2. QR code ───────────────────────────────────────────────────────────────
def build_qr():
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(GITHUB_PAGES_URL)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="white")
    img.save(QR_OUT)
    print(f"[+] QR code -> {QR_OUT}")
    return QR_OUT

# ── 3. Rebuild static figure with embedded QR stamp ─────────────────────────
def build_static(df, qr_path):
    df_cat = df[df["Model"] == "CatBoost"].dropna(subset=["PR-AUC"])
    df_tab = df[df["Model"] == "TabNet(LS+Pruning)"].dropna(subset=["PR-AUC"])

    view_angles = [("Left", 25, -45), ("Front", 25, 45), ("Right", 25, 135)]
    fig = plt.figure(figsize=(18, 6.5))

    for col, (label, elev, azim) in enumerate(view_angles):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")

        if not df_cat.empty:
            s1 = ax.plot_trisurf(
                df_cat["Anchor_Year"], df_cat["Evaluate_Year"], df_cat["PR-AUC"],
                cmap="Blues", alpha=0.82, edgecolor="steelblue", linewidth=0.15,
            )
            s1._facecolors2d = s1._facecolor3d
            s1._edgecolors2d = s1._edgecolor3d

        if not df_tab.empty:
            s2 = ax.plot_trisurf(
                df_tab["Anchor_Year"], df_tab["Evaluate_Year"], df_tab["PR-AUC"],
                cmap="Reds", alpha=0.60, edgecolor="firebrick", linewidth=0.15,
            )
            s2._facecolors2d = s2._facecolor3d
            s2._edgecolors2d = s2._edgecolor3d

        ax.set_xlabel("Anchor Year",  labelpad=8, fontsize=8)
        ax.set_ylabel("Horizon Year", labelpad=8, fontsize=8)
        ax.set_zlabel("PR-AUC",       labelpad=8, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {col+1}/3  ({label})", fontsize=9, pad=4)

    # Legend patches
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color="steelblue", alpha=0.82, label="CatBoost — Structural Guardrail")
    red_patch  = mpatches.Patch(color="firebrick", alpha=0.60, label="TabNet (LS+Pruning) — Deep Interpolation")
    fig.legend(
        handles=[blue_patch, red_patch],
        loc="lower center", ncol=2,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.42, -0.02),
    )

    plt.suptitle(
        "3D Temporal Drift Topology: Structural Guardrails vs Deep Interpolation",
        fontsize=14, y=1.01,
    )

    # ── QR stamp in bottom-right ────────────────────────────────────────────
    # Place a small inset axes in figure coordinates [left, bottom, width, height]
    qr_img  = Image.open(qr_path).convert("RGB")
    qr_size = 0.10   # fraction of figure
    ax_qr = fig.add_axes([0.88, -0.06, qr_size, qr_size * (18/6.5)])
    ax_qr.imshow(qr_img)
    ax_qr.axis("off")
    ax_qr.set_title("Scan for\ninteractive",
                     fontsize=6.5, pad=2, color="#444444")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(STATIC_OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Static 3-panel (with QR) -> {STATIC_OUT}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_df()
    if df is None:
        sys.exit(1)

    build_html(df)
    qr_path = build_qr()
    build_static(df, qr_path)
    print("[OK] All 3D outputs written.")

if __name__ == "__main__":
    main()
