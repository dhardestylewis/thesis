import pandas as pd
from sklearn.metrics import average_precision_score

df = pd.read_csv('Analysis/Output/Track1_Predictive/Metrics/stage_c_oof_predictions_H0.csv')
y_true = df['y_true']

print('--- Out-Of-Fold Cross-Validation PR-AUC ---')
print(f"CatBoost Optimal: {average_precision_score(y_true, df['y_prob']):.4f}")
print(f"Linear Regression: {average_precision_score(y_true, df['y_prob_lr']):.4f}")
print(f"Random Forest: {average_precision_score(y_true, df['y_prob_rf']):.4f}")
print(f"Non-Linear Anchor (CatBoost): {average_precision_score(y_true, df['y_prob_anchor']):.4f}")
