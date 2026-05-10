import pandas as pd
import numpy as np
import os

csv_path = r'C:\Users\dhl\data\Thesis\thesis\Analysis\Output\universal_ltr_matrix.csv'
df = pd.read_csv(csv_path)

# Calculate Mean, Min, Max, and 95% CI (1.96 * standard error)
agg = df.groupby('Profile').agg(
    Mean_NDCG=('NDCG', 'mean'),
    Min_NDCG=('NDCG', 'min'),
    Max_NDCG=('NDCG', 'max'),
    Std_NDCG=('NDCG', 'std'),
    Count=('NDCG', 'count')
).reset_index()

agg['CI_95'] = 1.96 * (agg['Std_NDCG'] / np.sqrt(agg['Count']))
agg['Lower_CI'] = agg['Mean_NDCG'] - agg['CI_95']
agg['Upper_CI'] = agg['Mean_NDCG'] + agg['CI_95']

# Format for output
agg['Mean'] = agg['Mean_NDCG'].round(3)
agg['Range'] = agg['Min_NDCG'].round(3).astype(str) + " - " + agg['Max_NDCG'].round(3).astype(str)
agg['95% CI'] = agg['Lower_CI'].round(3).astype(str) + " - " + agg['Upper_CI'].round(3).astype(str)

print("\n=== Universal LTR Stability Baseline (Across all Anchors & Years) ===")
final_display = agg[['Profile', 'Mean', 'Range', '95% CI']].sort_values('Mean', ascending=False)
print(final_display.to_markdown(index=False))

