"""
08h_update_multihorizon_latex_table.py

Reads the multi-model OOT CSVs and regenerates:
  Tables/chapter4_performance/tbl_ch4_09_multi_horizon_results.tex

Table layout (biweekly):
  Rows = Horizon (14-day, 3-month, 6-month, 1-year, 2-year)
  Cols = Model Family / Model, showing mean PR-AUC ± std and mean ROC-AUC
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
TABLES    = ROOT / "Thesis_Draft/Draft_v1/Tables/chapter4_performance"
TABLES.mkdir(parents=True, exist_ok=True)

BW_CSV  = ARTIFACTS / "multihorizon_multicutoff_all_models.csv"
ANN_CSV = ARTIFACTS / "annualized_multihorizon_multicutoff_all_models.csv"

HORIZON_ORDER_BW  = ["14_Days", "3_Months", "6_Months", "1_Year", "2_Years"]
HORIZON_LABELS_BW = {
    "14_Days":   "14-Day",
    "3_Months":  "3-Month",
    "6_Months":  "6-Month",
    "1_Year":    "1-Year",
    "2_Years":   "2-Year",
}

MODEL_FAMILY = {
    "CatBoost":    "Tree",
    "RandomForest":"Tree",
    "LogisticL2":  "Linear",
    "LogisticL1":  "Linear",
    "MLP":         "Deep",
}

FAMILY_ORDER = ["Tree", "Linear", "Deep"]
FAMILY_LABEL = {"Tree": "Tree Ensembles", "Linear": "Regularized Linear", "Deep": "Deep (MLP)"}


def build_table(df, horizon_order, horizon_labels, tag, caption, label):
    df = df.copy()
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizon_order, ordered=True)
    df["Model_Family"] = df["Model"].map(MODEL_FAMILY).fillna(df.get("Model_Family", "Other"))

    agg = (df.groupby(["Model_Family", "Model", "Horizon"])
             .agg(
                 PR_mean=("PR_AUC", "mean"),
                 PR_std=("PR_AUC", "std"),
                 ROC_mean=("ROC_AUC", "mean"),
                 Naive_mean=("Naive_PR_AUC", "mean"),
             )
             .reset_index()
          )
    agg["Lift_mean"] = agg["PR_mean"] / agg["Naive_mean"].clip(lower=1e-6)

    horizons = horizon_order

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption[Multi-Horizon OOT Performance by Architecture]" + "{" + caption + "}")
    lines.append(r"\label{" + label + "}")

    ncols = 2 + len(horizons) * 2  # Model | Family | (PR ± std, ROC) × horizons
    col_spec = "ll" + "cc" * len(horizons)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1
    h1 = r"Family & Model"
    for h in horizons:
        lbl = horizon_labels.get(h, h)
        h1 += r" & \multicolumn{2}{c}{" + lbl + "}"
    lines.append(h1 + r" \\")
    lines.append(r"\cmidrule(lr){3-4}" + "".join(
        r"\cmidrule(lr){" + str(3 + 2*i) + "-" + str(4 + 2*i) + "}"
        for i in range(1, len(horizons))
    ))

    # Header row 2
    h2 = r" & "
    for _ in horizons:
        h2 += r" & PR-AUC & ROC-AUC"
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    for family in FAMILY_ORDER:
        fam_sub = agg[agg["Model_Family"] == family]
        if fam_sub.empty:
            continue
        models = sorted(fam_sub["Model"].unique())
        first = True
        for model in models:
            m_sub = fam_sub[fam_sub["Model"] == model].set_index("Horizon").reindex(horizons)
            fam_label = FAMILY_LABEL.get(family, family) if first else ""
            row = fam_label + " & " + model
            for h in horizons:
                pr  = m_sub.loc[h, "PR_mean"] if h in m_sub.index else float("nan")
                roc = m_sub.loc[h, "ROC_mean"] if h in m_sub.index else float("nan")
                pr_s  = f"{pr:.3f}"  if not np.isnan(pr)  else "--"
                roc_s = f"{roc:.3f}" if not np.isnan(roc) else "--"
                row += f" & {pr_s} & {roc_s}"
            lines.append(row + r" \\")
            first = False
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{\textwidth}")
    lines.append(r"\vspace{2pt}\footnotesize")
    lines.append(r"\textit{Note:} Mean PR-AUC and ROC-AUC across walk-forward test years. "
                 r"PR-AUC is the primary metric given class imbalance; ROC-AUC shown for reference. "
                 r"Tree = CatBoost + Random Forest; Linear = Logistic L1/L2; Deep = MLP. ")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def run():
    for csv_path, horizon_order, horizon_labels, tag, caption, label in [
        (
            BW_CSV,
            HORIZON_ORDER_BW,
            HORIZON_LABELS_BW,
            "biweekly",
            "Multi-Horizon Walk-Forward OOT Performance by Architecture (Biweekly Panel)",
            "tbl:multihorizon_biweekly"
        ),
    ]:
        if not csv_path.exists():
            print(f"[SKIP] CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        tex = build_table(df, horizon_order, horizon_labels, tag, caption, label)
        out = TABLES / f"tbl_ch4_09_multi_horizon_results_{tag}.tex"
        out.write_text(tex)
        print(f"[+] Written: {out.name}")

    print("[+] LaTeX table update complete.")


if __name__ == "__main__":
    run()
