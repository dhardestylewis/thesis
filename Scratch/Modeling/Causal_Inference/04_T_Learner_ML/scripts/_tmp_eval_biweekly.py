import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"

FEATS_DYNAMIC = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum",
    "treasury_10yr_yield", "fed_funds_rate", "local_unemployment_rate",
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

print("Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)
df_hazard = df[df["period_seq"] > 0].copy()
target = "petition_event"

model_df = df_hazard[["case_number", target] + FEATS_DYNAMIC].copy().dropna()

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))

train = model_df.iloc[train_idx]
test = model_df.iloc[test_idx]

X_train, y_train = train[FEATS_DYNAMIC], train[target]
X_test, y_test = test[FEATS_DYNAMIC], test[target]

pos_count = y_train.sum()
scale_pos_weight = (len(y_train) - pos_count) / pos_count if pos_count > 0 else 1.0

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    scale_pos_weight=scale_pos_weight,
    eval_metric='PRAUC',
    random_seed=42,
    verbose=0,
    task_type="GPU"
)

model.fit(Pool(X_train, y_train))

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
cr = classification_report(y_test, y_pred)

print(f"\n--- PERFORMANCE: NIMBY Hazard (Biweekly Panel) ---")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC:  {pr_auc:.4f}")
print(f"\nClassification Report:\n{cr}")
