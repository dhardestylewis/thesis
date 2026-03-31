import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os

def run_stage_b():
    df = pd.read_csv(r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv', on_bad_lines='skip', engine='python')
    df = df.dropna(subset=['delta_max_far', 'gross_site_area_acres', 'year'])

    def derive_6_tier(row):
        far = row['delta_max_far']
        acres = row['gross_site_area_acres']
        if acres > 3 and far > 1.5: return "PUD / large negotiated project"
        if far > 1.0: return "mixed-use"
        if far > 0.5: return "multifamily"
        if far > 0.1 and acres < 1.0: return "missing-middle"
        if far > 0: return "discretionary rezoning"
        return "by-right infill"

    df['6_tier_class'] = df.apply(derive_6_tier, axis=1)

    X = df[['gross_site_area_acres', 'year']]
    y = df['6_tier_class']

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    cb = CatBoostClassifier(iterations=60, depth=4, learning_rate=0.08, loss_function='MultiClass', verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=15)

    preds = cb.predict(X).flatten()
    f1 = f1_score(y, preds, average='macro', zero_division=0)
    print(f"Macro-F1 (6-Tier): {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y, preds, zero_division=0))

    out_path = os.path.join(r'C:\Users\dhl\data\thesis\thesis\Analysis\Output\Track1_Predictive', 'stage_b_model.cbm')
    cb.save_model(out_path)
    print(f"Saved Stage B model to {out_path}")

if __name__ == '__main__':
    run_stage_b()
