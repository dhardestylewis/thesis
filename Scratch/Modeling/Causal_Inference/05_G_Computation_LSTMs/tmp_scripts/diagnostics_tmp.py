import os
import pandas as pd
import causal_cfm_cvae

print("--- DIAGNOSTICS ---")

# Data diagnostics
X, Y, L, features, targets, norm_dict, treat_idx, cases, cell_assignments, filing_years = causal_cfm_cvae.load_data()

print("cumulative_petition_pct_lag1 in features:", "cumulative_petition_pct_lag1" in features)
print("petition features:", [f for f in features if "petition" in f])

df = pd.read_csv(causal_cfm_cvae.PANEL_PATH)
def _fraction_01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if x.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(lower=0.0, upper=1.0)

if "petition_pct_this_period" not in df.columns:
    df["petition_pct_this_period"] = 0.0
df["petition_pct_this_period"] = _fraction_01(df["petition_pct_this_period"])

df["cumulative_petition_pct"] = (
    df.groupby("case_number")["petition_pct_this_period"]
      .transform(lambda s: s.fillna(0.0).cumsum())
      .clip(lower=0.0, upper=1.0)
)

print("\ndf['petition_pct_this_period'].describe():")
print(df["petition_pct_this_period"].describe())
print("\ndf['cumulative_petition_pct'].describe():")
print(df["cumulative_petition_pct"].describe())
