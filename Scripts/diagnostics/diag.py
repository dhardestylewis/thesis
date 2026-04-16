import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os

print('--- DIAGNOSING STAGE D SURVIVORSHIP BIAS ---')
df_h0 = pd.read_csv('Data/Warehouse_As_Of/canonical/H0_Filing_Master_Enriched_v2.csv', low_memory=False)

opposed_cases = df_h0[df_h0['is_protested'] == 1].copy()
print(f'Total originally opposed cases in H0: {len(opposed_cases)}')

if 'case_status' in opposed_cases.columns:
    print('\nDisposition of opposed cases (case_status):')
    print(opposed_cases['case_status'].value_counts(dropna=False).head(10))

if 'withdrawn' in opposed_cases.columns:
    print('\nWithdrawn flag for opposed cases:')
    print(opposed_cases['withdrawn'].value_counts(dropna=False))

votes = pd.read_csv('Data/Zoning_Cases/Processed_Data/CSV/submission_grade_goldmine_tensor.csv', usecols=['CASE_NUMBER', 'vote_yes', 'vote_no'])

merged = opposed_cases.merge(votes, left_on='case_number', right_on='CASE_NUMBER', how='left')
print(f'\nTotal opposed cases successfully merged with a council vote record: {merged["vote_yes"].notna().sum()}')
print(f'Total opposed cases MISSING a council vote record (likely withdrawn/stalled): {merged["vote_yes"].isna().sum()}')


print('\n--- DIAGNOSING 0.000 FNR GAP AND FPR BALLOONING ---')
# Simulate Stage C's exact cross validation on H0 to look at the empirical probabilities
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

X = df_h0.select_dtypes(include=[np.number]).fillna(0)
X = X.drop(columns=['is_protested', 'year', 'council_district'], errors='ignore')
y = df_h0['is_protested'].fillna(0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cb = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42)
calibrated_cb = CalibratedClassifierCV(estimator=cb, method='isotonic', cv=3)
calibrated_cb.fit(X_train, y_train)

preds = calibrated_cb.predict_proba(X_test)[:, 1]
threshold = y_train.mean()

print(f'\nEmpirical Threshold (y_train.mean()): {threshold:.4f}')

y_pred_bin = (preds > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_bin)
print(f'\nConfusion Matrix at threshold {threshold:.4f}:')
print(cm)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
print(f'False Positive Rate (FPR): {fpr:.4f}')
print(f'False Negative Rate (FNR): {fnr:.4f}')

print(f'\nMean predicted probability for True Negatives: {np.mean(preds[y_test == 0]):.4f}')
print(f'Mean predicted probability for True Positives: {np.mean(preds[y_test == 1]):.4f}')
print(f'Max predicted probability overall: {np.max(preds):.4f}')
