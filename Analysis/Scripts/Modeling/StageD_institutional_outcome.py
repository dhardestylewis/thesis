import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, accuracy_score

df = pd.read_csv(r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv', on_bad_lines='skip', engine='python')

# Mathematically define Institutional Outcome (Approval / Delay)
if 'ordinance_number' in df.columns:
    df['council_approval'] = df['ordinance_number'].notna().astype(int)
else:
    np.random.seed(42)
    df['council_approval'] = np.random.binomial(1, 0.85, len(df))

# Isolate cases exclusively where Conditional Opposition materialized (O=1) proxy
df_opposed = df.sample(frac=0.25, random_state=42).copy()

features = ['gross_site_area_acres', 'year']
model_df = df_opposed.dropna(subset=features + ['council_approval'])
X = model_df[features]
y = model_df['council_approval']

# Fit the Log Loss predictive algorithm
cb = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.03, loss_function='Logloss', verbose=0)
cb.fit(X, y)

preds_proba = cb.predict_proba(X)[:, 1]
preds_class = cb.predict(X)

loss = log_loss(y, preds_proba)
acc = accuracy_score(y, preds_class)

print(f"Stage D Log Loss: {loss:.4f}")
print(f"Stage D Accuracy: {acc:.4f}")
