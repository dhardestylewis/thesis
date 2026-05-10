import pandas as pd
import numpy as np

def build_target(df: pd.DataFrame, window: int) -> pd.Series:
    if window == 1:
        return df["petition_event"].astype(int)
    target = df.groupby("case_number")["petition_event"].transform(
        lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1]
    )
    return target.fillna(0).astype(int)

print("Loading biweekly panel...")
df_raw = pd.read_csv('Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv')

# Truncate post-petition rows
mask = (df_raw.groupby('case_number')['cumulative_petition_events'].shift(1).fillna(0) == 0)
df = df_raw[mask].copy()

# Ensure chronological sort
df['period_start'] = pd.to_datetime(df['period_start'])
df = df.sort_values(by=['case_number', 'period_seq'])

print(f"\nFiltered dataset contains {len(df)} rows across {df['case_number'].nunique()} cases.")
print(f"Total precise EDIMS OCR petitions injected: {df_raw['petition_event'].sum()}")

horizons = {
    '14_Days': 1,
    '3_Months': 6,
    '6_Months': 13,
    '1_Year': 26,
    '2_Years': 52
}

print("\n" + "="*40)
print("TARGET DISTRIBUTION (POSITIVE CLASS %)")
print("="*40)

for h_name, window in horizons.items():
    target = build_target(df, window)
    positives = target.sum()
    total = len(target)
    pct = (positives / total) * 100
    
    print(f"[{h_name:<10}] Positives: {positives:<5} | Total: {total:<7} | Rate: {pct:.2f}%")

print("\n" + "="*40)
print("TARGET SPARSITY BY YEAR (2_Years Horizon)")
print("="*40)

target_2y = build_target(df, 52)
df['Target_2Y'] = target_2y

for year in sorted(df['year'].unique()):
    year_df = df[df['year'] == year]
    pos = year_df['Target_2Y'].sum()
    pct = (pos / len(year_df)) * 100 if len(year_df) > 0 else 0
    print(f"[{year}] Positives: {pos:<4} | Rate: {pct:.2f}%")
