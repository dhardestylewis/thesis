"""
Track 2: Regression Discontinuity at the 20% petition threshold.
Running variable: label_petition_total_pct (sum of area_pct for signers)
Treatment:        label_valid_protest (label_petition_total_pct >= 20)
Outcomes:         zoning_changed, direction, periods_after_petition
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH  = rf"{OUT_DIR}\biweekly_panel.csv"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
REC_PATH    = rf"{OUT_DIR}\recovered_petitions.csv"

panel  = pd.read_csv(PANEL_PATH, low_memory=False)
master = pd.read_csv(MASTER_PATH, low_memory=False)
rec    = pd.read_csv(REC_PATH)

# ── Build per-case petition summary ─────────────────────────────────────────
pet_by_case = rec.groupby("case_number").agg(
    petition_n_parcels=("tcad_id", "count"),
    petition_area_pct_raw=("area_pct", "sum"),
).reset_index()
pet_by_case["label_valid_protest"] = (pet_by_case["petition_area_pct_raw"] >= 20).astype(int)
pet_by_case["running_var"]   = pet_by_case["petition_area_pct_raw"] - 20  # centered at cutoff

# Save petition intensity artifact
pet_by_case.to_csv(rf"{OUT_DIR}\petition_intensity.csv", index=False)
print(f"petition_intensity.csv saved: {len(pet_by_case)} cases")
print(f"  Valid protest (>=20%): {pet_by_case['label_valid_protest'].sum()}")
print(f"  Below threshold:       {(~pet_by_case['label_valid_protest'].astype(bool)).sum()}")

# ── Attach outcomes from master ──────────────────────────────────────────────
OVERLAY_STRIP = __import__("re").compile(
    r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)"
)
INTENSITY = {
    "W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,
    "SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,
    "MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,
    "LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,
    "LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6,
}
def base_zone(z):
    if not isinstance(z, str): return None
    z = OVERLAY_STRIP.sub("", z.strip().upper()).strip("-")
    return z
def intensity(z):
    return INTENSITY.get(base_zone(z), np.nan)

outcomes = master[["case_number","Requested_Zoning","Final_Zoning"]].dropna(
    subset=["Requested_Zoning","Final_Zoning"]).drop_duplicates("case_number")

# Terminal period per case
panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce")
terminal = (panel.sort_values("period_seq").groupby("case_number").agg(
    terminal_seq=("period_seq","max")).reset_index())
petition_period = (panel[panel["petition_event"]==1]
    .sort_values("period_seq").groupby("case_number").first()
    .reset_index()[["case_number","period_seq"]]
    .rename(columns={"period_seq":"petition_seq"}))
terminal = terminal.merge(petition_period, on="case_number", how="left")
terminal["periods_after_petition"] = terminal["terminal_seq"] - terminal["petition_seq"]

# Merge all
df = pet_by_case.merge(outcomes, on="case_number", how="left")
df = df.merge(terminal, on="case_number", how="left")
df["req_intensity"] = df["Requested_Zoning"].apply(intensity)
df["fin_intensity"] = df["Final_Zoning"].apply(intensity)
df["zoning_changed"] = (df["Requested_Zoning"].str.strip() != df["Final_Zoning"].str.strip()).astype(float)
df["downgrade"] = ((df["fin_intensity"] < df["req_intensity"]) & df["zoning_changed"].astype(bool)).astype(float)
df["upgrade"]   = ((df["fin_intensity"] > df["req_intensity"]) & df["zoning_changed"].astype(bool)).astype(float)

rd_df = df.dropna(subset=["running_var"])
print(f"\nRD sample: {len(rd_df)} cases with running variable")
print(f"  With outcome data: {rd_df['zoning_changed'].notna().sum()}")

# ── McCrary density test (manual histogram check) ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Regression Discontinuity: Protest Petition 20% Threshold\n(Running Variable: Total Signed Area %  |  Treatment: Valid Protest)",
             color="white", fontsize=12, fontweight="bold", y=1.01)

# Panel 1: Density of running variable (McCrary check)
ax = axes[0, 0]
bins = np.linspace(-80, 200, 40)
left  = rd_df[rd_df["running_var"] < 0]["running_var"]
right = rd_df[rd_df["running_var"] >= 0]["running_var"]
ax.hist(left,  bins=bins, color="#60a5fa", alpha=0.7, label="Below threshold")
ax.hist(right, bins=bins, color="#f97316", alpha=0.7, label="Above threshold")
ax.axvline(0, color="white", linestyle="--", linewidth=1.5)
ax.set_title("McCrary Density Check", color="white", fontsize=10)
ax.set_xlabel("Running variable (area_pct - 20)", color="white")
ax.set_ylabel("Count", color="white")
ax.tick_params(colors="white")
ax.set_facecolor("#1a1a2e")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# Panels 2-4: Local linear regression for each outcome
OUTCOMES = [
    ("zoning_changed", "Zoning Changed (Requested!=Final)", "#a78bfa"),
    ("downgrade",      "Downgrade Rate",                   "#34d399"),
    ("upgrade",        "Upgrade Rate",                     "#f43f5e"),
]

def local_linear(data, outcome, bandwidths=[10, 20, 40]):
    results = []
    for bw in bandwidths:
        sub = data[(data["running_var"] >= -bw) & (data["running_var"] <= bw)].dropna(subset=[outcome])
        if len(sub) < 10:
            results.append({"bw": bw, "ate": np.nan, "se": np.nan, "n": len(sub)})
            continue
        left  = sub[sub["running_var"] < 0]
        right = sub[sub["running_var"] >= 0]
        # Local mean on each side
        left_mean  = left[outcome].mean()
        right_mean = right[outcome].mean()
        # SE via bootstrap
        pooled_se = np.sqrt(left[outcome].var()/max(len(left),1) + right[outcome].var()/max(len(right),1))
        results.append({"bw": bw, "ate": right_mean - left_mean,
                        "se": pooled_se, "n": len(sub),
                        "left_mean": left_mean, "right_mean": right_mean})
    return results

for i, (col, title, color) in enumerate(OUTCOMES):
    ax = axes[0, i+1] if i < 2 else axes[1, 0]
    sub = rd_df.dropna(subset=[col])
    if len(sub) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color="white", transform=ax.transAxes)
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white", fontsize=9)
        continue
    # Scatter + local means by bin
    bin_edges = np.arange(-80, 200, 20)
    bin_mids, bin_means, bin_ns = [], [], []
    for j in range(len(bin_edges)-1):
        mask = (sub["running_var"] >= bin_edges[j]) & (sub["running_var"] < bin_edges[j+1])
        if mask.sum() > 0:
            bin_mids.append((bin_edges[j]+bin_edges[j+1])/2)
            bin_means.append(sub[mask][col].mean())
            bin_ns.append(mask.sum())
    left_bins  = [(m, v, n) for m, v, n in zip(bin_mids, bin_means, bin_ns) if m < 0]
    right_bins = [(m, v, n) for m, v, n in zip(bin_mids, bin_means, bin_ns) if m >= 0]
    if left_bins:
        ax.scatter([x[0] for x in left_bins], [x[1] for x in left_bins],
                   s=[max(x[2]*8,20) for x in left_bins], color="#60a5fa", alpha=0.8, zorder=3)
        xs = [x[0] for x in left_bins]; ys = [x[1] for x in left_bins]
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            xf = np.linspace(min(xs), 0, 50)
            ax.plot(xf, m*xf+b, color="#60a5fa", linewidth=2)
    if right_bins:
        ax.scatter([x[0] for x in right_bins], [x[1] for x in right_bins],
                   s=[max(x[2]*8,20) for x in right_bins], color=color, alpha=0.8, zorder=3)
        xs = [x[0] for x in right_bins]; ys = [x[1] for x in right_bins]
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            xf = np.linspace(0, max(xs), 50)
            ax.plot(xf, m*xf+b, color=color, linewidth=2)
    ax.axvline(0, color="white", linestyle="--", linewidth=1.5)
    res = local_linear(rd_df, col)
    ate_strs = [f"bw={r['bw']}: ATE={r['ate']:+.3f} (n={r['n']})" for r in res if not np.isnan(r.get("ate", np.nan))]
    ax.set_title(f"{title}\n" + " | ".join(ate_strs[:2]), color="white", fontsize=8)
    ax.set_xlabel("Running variable (area_pct - 20)", color="white", fontsize=8)
    ax.set_ylabel(col, color="white", fontsize=8)
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a2e")

# Panel 5: Summary table
ax = axes[1, 1]
ax.set_facecolor("#1a1a2e")
ax.set_axis_off()
summary_rows = []
for col, title, _ in OUTCOMES:
    res = local_linear(rd_df, col)
    for r in res:
        if not np.isnan(r.get("ate", np.nan)):
            summary_rows.append([title[:25], f"bw={r['bw']}", f"n={r['n']}",
                                  f"{r.get('left_mean',np.nan):.3f}",
                                  f"{r.get('right_mean',np.nan):.3f}",
                                  f"{r['ate']:+.3f}"])
if summary_rows:
    tbl = ax.table(
        cellText=summary_rows,
        colLabels=["Outcome","BW","N","Below 20%","Above 20%","ATE"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1a2e" if r > 0 else "#2d2d4e")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#444")
ax.set_title("RD Summary Table", color="white", fontsize=10, pad=10)

# Panel 6: N parcels vs running variable
ax = axes[1, 2]
ax.scatter(rd_df["running_var"], rd_df["petition_n_parcels"],
           c=rd_df["label_valid_protest"].map({0:"#60a5fa",1:"#f97316"}),
           alpha=0.6, s=25, edgecolors="none")
ax.axvline(0, color="white", linestyle="--", linewidth=1.5)
ax.set_title("Petition Parcel Count vs Running Variable", color="white", fontsize=9)
ax.set_xlabel("Running variable (area_pct - 20)", color="white", fontsize=8)
ax.set_ylabel("N signing parcels", color="white", fontsize=8)
ax.tick_params(colors="white")
ax.set_facecolor("#1a1a2e")
ax.set_xlim(-80, 200)

plt.tight_layout()
out_path = rf"{OUT_DIR}\rd_protest_threshold.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
print(f"\nRD plot saved: {out_path}")

# ── Print RD summary ─────────────────────────────────────────────────────────
print("\n=== RD RESULTS SUMMARY ===")
for col, title, _ in OUTCOMES:
    res = local_linear(rd_df, col)
    print(f"\n{title}:")
    for r in res:
        if "left_mean" in r and not np.isnan(r["ate"]):
            z_stat = r["ate"] / (r["se"] + 1e-9)
            p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            print(f"  bw={r['bw']:3d}: below={r['left_mean']:.3f} above={r['right_mean']:.3f} "
                  f"ATE={r['ate']:+.3f} SE={r['se']:.3f} z={z_stat:.2f} p={p_val:.3f} n={r['n']}")
