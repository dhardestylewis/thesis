"""
rebuild_nlp_features_temporal.py
----------------------------------
Replaces the static case-level NLP feature merge with a temporally-anchored
cumulative merge using nlp_event_log.csv.

The existing case_nlp_features.csv aggregates ALL documents for a case into a
single row, meaning early panel periods receive keyword counts from documents
that hadn't been written yet -- a leakage of 81,234 rows.

Fix: for each panel row at period_start, sum only the NLP events whose
event_date < period_start. This is computed via a running cumsum per
(case_number, source), then joined to the panel with merge_asof(backward).

New columns (replace existing):
  commission source:
    nlp_document_count        -- # commission docs seen so far
    nlp_total_tokens          -- cumulative token count
    nlp_oppose_hits           -- cumulative opposition keyword hits
    nlp_traffic_hits          -- cumulative traffic keyword hits
    nlp_density_hits          -- cumulative density keyword hits
  council source:
    council_nlp_document_count
    council_nlp_total_tokens
    council_nlp_oppose_hits
    council_nlp_traffic_hits
    council_nlp_density_hits

Leakage contract: merge_asof(direction='backward') on event_date <= period_start.
"""
import pandas as pd
import numpy as np

BASE      = r"c:\Users\dhl\data\Thesis\thesis\Data"
LOG_PATH  = BASE + r"\interim\nlp_event_log.csv"
PANEL_IN  = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
PANEL_OUT = PANEL_IN

NLP_COLS_COMM  = ["nlp_document_count", "nlp_total_tokens",
                  "nlp_oppose_hits", "nlp_traffic_hits", "nlp_density_hits"]
NLP_COLS_COUNC = ["council_nlp_document_count", "council_nlp_total_tokens",
                  "council_nlp_oppose_hits", "council_nlp_traffic_hits",
                  "council_nlp_density_hits"]

print("Loading NLP event log...", flush=True)
log = pd.read_csv(LOG_PATH, low_memory=False)
log["event_date"] = pd.to_datetime(log["event_date"], errors="coerce")
log = log.dropna(subset=["event_date"]).sort_values(["case_number", "event_date"])
print("  {:,} events | {:,} cases | sources: {}".format(
    len(log), log["case_number"].nunique(), log["source"].unique().tolist()))

# ------------------------------------------------------------------
# Build cumulative timeline per (case_number, source)
# Each row = cumulative sum at that event_date
# ------------------------------------------------------------------
def build_cumulative(src_label, col_prefix):
    """
    Returns a DataFrame of cumulative NLP stats at each event date,
    for a given source ('commission' or 'council').
    """
    src = log[log["source"] == src_label].copy()
    src = src.sort_values(["case_number", "event_date"])

    # One row per (case_number, event_date) -- sum if multiple docs on same day
    daily = (
        src.groupby(["case_number", "event_date"])
           .agg(doc_count=("tokens", "count"),
                tokens=("tokens", "sum"),
                oppose=("oppose", "sum"),
                traffic=("traffic", "sum"),
                density=("density", "sum"))
           .reset_index()
           .sort_values(["case_number", "event_date"])
    )

    # Cumulative sum within each case
    for col in ["doc_count", "tokens", "oppose", "traffic", "density"]:
        daily["cum_" + col] = daily.groupby("case_number")[col].cumsum()

    # Rename to final panel column names
    daily = daily.rename(columns={
        "cum_doc_count": col_prefix + "_document_count",
        "cum_tokens":    col_prefix + "_total_tokens",
        "cum_oppose":    col_prefix + "_oppose_hits",
        "cum_traffic":   col_prefix + "_traffic_hits",
        "cum_density":   col_prefix + "_density_hits",
    })

    keep = ["case_number", "event_date",
            col_prefix + "_document_count",
            col_prefix + "_total_tokens",
            col_prefix + "_oppose_hits",
            col_prefix + "_traffic_hits",
            col_prefix + "_density_hits"]
    return daily[keep].sort_values("event_date")

print("Building commission cumulative timeline...", flush=True)
comm_cum = build_cumulative("commission", "nlp")
print("  {:,} rows | {:,} cases".format(len(comm_cum), comm_cum["case_number"].nunique()))

print("Building council cumulative timeline...", flush=True)
counc_cum = build_cumulative("council", "council_nlp")
print("  {:,} rows | {:,} cases".format(len(counc_cum), counc_cum["case_number"].nunique()))

# ------------------------------------------------------------------
# Load panel and drop old (static/leaked) NLP columns
# ------------------------------------------------------------------
print("\nLoading panel...", flush=True)
panel = pd.read_csv(PANEL_IN, low_memory=False)
panel["period_start"] = pd.to_datetime(panel["period_start"])
n_start = len(panel)
print("  {:,} rows | {:,} cases".format(n_start, panel["case_number"].nunique()))

# Drop all existing NLP columns (both commission and council)
all_nlp_cols = NLP_COLS_COMM + NLP_COLS_COUNC
panel = panel.drop(columns=[c for c in all_nlp_cols if c in panel.columns], errors="ignore")

# ------------------------------------------------------------------
# merge_asof: for each panel row, take cumulative sum of all NLP
# events with event_date <= period_start
# ------------------------------------------------------------------
def asof_merge_nlp(panel_df, cum_df, cols, label):
    """merge_asof backward: at period_start, use the last cumulative state."""
    panel_sorted = panel_df.sort_values("period_start")
    cum_sorted   = cum_df.sort_values("event_date")

    merged = pd.merge_asof(
        panel_sorted,
        cum_sorted[["case_number", "event_date"] + cols],
        left_on="period_start",
        right_on="event_date",
        by="case_number",
        direction="backward",
    ).drop(columns=["event_date"], errors="ignore")

    # Fill NaN with 0 -- periods before any NLP document had zero hits
    for col in cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    n_nonzero = (merged[cols[0]] > 0).sum()
    print("  [{label}] {col}: {n:,} non-zero rows".format(
        label=label, col=cols[0], n=n_nonzero))
    return merged

print("\nMerging commission NLP features (leakage-safe)...", flush=True)
panel = asof_merge_nlp(panel, comm_cum, NLP_COLS_COMM, "commission")

print("Merging council NLP features (leakage-safe)...", flush=True)
panel = asof_merge_nlp(panel, counc_cum, NLP_COLS_COUNC, "council")

assert len(panel) == n_start, "Row count changed!"

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
panel.to_csv(PANEL_OUT, index=False)
print("\nSaved: {}".format(PANEL_OUT))
print("Shape: {:,} rows x {} cols".format(len(panel), len(panel.columns)))

# Quick leakage spot-check: find any case with NLP hits before its first event_date
print("\nSpot-checking leakage...", flush=True)
first_event = log.groupby("case_number")["event_date"].min().reset_index()
first_event.columns = ["case_number", "first_event_date"]
check = panel.merge(first_event, on="case_number", how="left")
check["first_event_date"] = pd.to_datetime(check["first_event_date"])
violations = check[
    (check["nlp_oppose_hits"] > 0) &
    check["first_event_date"].notna() &
    (check["period_start"] < check["first_event_date"])
]
print("  Leakage violations (nlp_oppose_hits > 0 before first event): {:,}".format(len(violations)))
if len(violations) == 0:
    print("  PASS: No pre-event NLP hits found.")
else:
    print("  FAIL: Violations detected:")
    print(violations[["case_number", "period_start", "first_event_date", "nlp_oppose_hits"]].head(5).to_string())
