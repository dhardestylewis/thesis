import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML")
from run_causal_ml_sweep import load_fully_hydrated_data

print("Loading Fully Hydrated Annualized Data Matrix...")
df, features, categorical_features = load_fully_hydrated_data()

thresh = 20
df = df.dropna(subset=["exact_geometric_petition_pct"])
df["target"] = (df["exact_geometric_petition_pct"] >= thresh).astype(int)

model_df = df[["case_number", "target"] + features].copy().dropna()

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))

train = model_df.iloc[train_idx]
test = model_df.iloc[test_idx]

X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

pos_count = y_train.sum()
scale_pos_weight = (len(y_train) - pos_count) / pos_count if pos_count > 0 else 1.0

model = CatBoostClassifier(
    iterations=300, 
    learning_rate=0.05, 
    depth=6, 
    verbose=0, 
    random_seed=42, 
    auto_class_weights='Balanced',
    task_type="GPU"
)

model.fit(Pool(X_train, y_train, cat_features=categorical_features))

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
cr = classification_report(y_test, y_pred)

print(f"\n--- PERFORMANCE: Static Protest (Annualized Panel) ---")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC:  {pr_auc:.4f}")
print(f"\nClassification Report:\n{cr}")
