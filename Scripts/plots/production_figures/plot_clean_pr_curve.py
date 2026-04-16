import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os, sys

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

sys.path.append(os.path.abspath('Scripts'))
try:
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT_DIR = os.path.abspath('.')
FIG_DIR = os.path.join(ROOT_DIR, 'Thesis_Draft', 'Draft_v1', 'Figures', 'exhibits')
os.makedirs(FIG_DIR, exist_ok=True)

class NonLinearAnchorRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=10.0, n_anchors=None):
        self.gamma = gamma
        self.n_anchors = n_anchors
        self.model = CatBoostRegressor(iterations=100, depth=5, random_seed=42, verbose=0)
        self.proj_X = LinearRegression(fit_intercept=False)
        self.proj_y = LinearRegression(fit_intercept=False)
        
    def fit(self, X_transformed, y, sample_weight=None):
        if isinstance(X_transformed, pd.DataFrame): X_transformed = X_transformed.values
        if isinstance(y, (pd.Series, pd.DataFrame)): y = y.values
        A = X_transformed[:, :self.n_anchors]
        self.proj_X.fit(A, X_transformed, sample_weight=sample_weight)
        self.proj_y.fit(A, y, sample_weight=sample_weight)
        X_P = self.proj_X.predict(A)
        y_P = self.proj_y.predict(A)
        factor = np.sqrt(self.gamma) - 1.0
        X_anc = X_transformed + factor * X_P
        y_anc = y + factor * y_P
        self.model.fit(X_anc, y_anc, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X_transformed):
        if isinstance(X_transformed, pd.DataFrame): X_transformed = X_transformed.values
        preds = self.model.predict(X_transformed)
        preds = np.clip(preds, 0, 1)
        return np.vstack([1 - preds, preds]).T

# 1. Load CatBoost native predictions directly from registry
reg_df = pd.read_parquet(os.path.join(ROOT_DIR, "registries", "prediction_registry.parquet"))
reg_df = reg_df[(reg_df["role"] == "test") & (reg_df["model_family"] == "CatBoost")].copy()
y_test_cb = reg_df["y_true"].astype(int).values
y_prob_cb = reg_df["y_score_calibrated"].astype(float).values

# 2. Simulate baseline models natively
labels = pd.read_parquet(os.path.join(ROOT_DIR, "registries", "label_registry.parquet"))
splits = pd.read_parquet(os.path.join(ROOT_DIR, "registries", "split_registry.parquet"))
features = pd.read_parquet(os.path.join(ROOT_DIR, "data", "interim", "stage_c_features_raw.parquet"))

labels = labels[labels["label_version"] == "label_v1_reconstructed_threshold_crossing"]
splits = splits[splits["split_id"] == "TEMP_OOD_2023_MAIN"]

df = features.merge(labels[["case_id", "threshold_crossed"]], on="case_id")
df = df.merge(splits[["case_id", "role"]], on="case_id")

# Create year_str using existing as_of_date directly on the dataframe
df['year_str'] = pd.to_datetime(df['as_of_date']).dt.year.astype(str)
df['council_district'] = df['council_district'].astype(str)

train_df = df[df["role"] != "test"].copy()
test_df = df[df["role"] == "test"].copy()

# Ensure we exclude the date columns so they are not fed into our standard ml stack
excluded = {"case_id", "threshold_crossed", "role", "as_of_date", "year_str"}
feature_cols = [c for c in train_df.columns if c not in excluded]

cat_cols = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]

# Core Imputation
imp = SimpleImputer(strategy='median')
train_num_imp = imp.fit_transform(train_df[num_cols])
test_num_imp = imp.transform(test_df[num_cols])

# Standard Scale
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_num_imp)
X_test_num = scaler.transform(test_num_imp)

y_train = train_df["threshold_crossed"].astype(int).values
y_test_baseline = test_df["threshold_crossed"].astype(int).values

y_probs = {}

# XGBoost
if XGBClassifier is not None:
    xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_num, y_train)
    y_probs["XGBoost"] = xgb.predict_proba(X_test_num)[:, 1]

# Logistic (L2)
lr = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=500, random_state=42)
lr.fit(X_train_num, y_train)
y_probs["Logistic (L2)"] = lr.predict_proba(X_test_num)[:, 1]

# Spatial-FE Logistic
sp_prep = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['council_district']),
    ('num', 'passthrough', num_cols)
])
X_train_sp = sp_prep.fit_transform(train_df)
X_test_sp = sp_prep.transform(test_df)

sp_imp = SimpleImputer(strategy='median')
sp_scaler = StandardScaler()
X_train_sp_sc = sp_scaler.fit_transform(sp_imp.fit_transform(X_train_sp))
X_test_sp_sc = sp_scaler.transform(sp_imp.transform(X_test_sp))

lr_sp = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=500, random_state=42)
lr_sp.fit(X_train_sp_sc, y_train)
y_probs["Spatial-FE Logistic"] = lr_sp.predict_proba(X_test_sp_sc)[:, 1]

# TabNet
if TabNetClassifier is not None:
    tab = TabNetClassifier(verbose=0)
    tab.fit(X_train_num, y_train, max_epochs=25, patience=5)
    y_probs["TabNet"] = tab.predict_proba(X_test_num)[:, 1]

# Anchor Regression (Causal)
anc_prep = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['council_district', 'year_str']),
    ('num', 'passthrough', num_cols)
])
X_train_anc = anc_prep.fit_transform(train_df)
X_test_anc = anc_prep.transform(test_df)

n_anc_dummies = len(anc_prep.named_transformers_['cat'].get_feature_names_out())
anc_imp = SimpleImputer(strategy='median')
anc_scaler = StandardScaler()

X_train_anc_sc = anc_scaler.fit_transform(anc_imp.fit_transform(X_train_anc))
X_test_anc_sc = anc_scaler.transform(anc_imp.transform(X_test_anc))

if CatBoostRegressor is not None:
    nonlin_anc = NonLinearAnchorRegression(n_anchors=n_anc_dummies)
    nonlin_anc.fit(X_train_anc_sc, y_train)
    y_probs["Anchor Regression (Causal)"] = nonlin_anc.predict_proba(X_test_anc_sc)[:, 1]

# Plot
plt.figure(figsize=(7, 6))

colors = {
    "Logistic (L2)": "gray",
    "Spatial-FE Logistic": "teal",
    "XGBoost": "green",
    "TabNet": "purple",
    "Anchor Regression (Causal)": "coral"
}

def plot_pr(y_t, y_p, label, color, style='-'):
    if y_p is not None:
        p, r, _ = precision_recall_curve(y_t, y_p)
        auc = average_precision_score(y_t, y_p)
        plt.plot(r, p, style, label=f"{label} (AUC={auc:.3f})", color=color, linewidth=2)

base_rate = y_test_cb.mean()

for model_name, probas in y_probs.items():
    plot_pr(y_test_baseline, probas, model_name, colors.get(model_name, "black"), style='--')

plot_pr(y_test_cb, y_prob_cb, "CatBoost Primary", "navy", style='-')

plt.axhline(base_rate, color='black', linestyle=':', label=f"Random Chance (AUC={base_rate:.3f})")

plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Positive Predictive Value)')

title_str = "Filing-Date Precision-Recall Curves by Model Class"
try:
    import json
    with open(os.path.join(ROOT_DIR, "Scripts", "exhibit_titles.json"), "r") as f:
        titles = json.load(f)
        if "track1_OOF_stage_c" in titles:
            title_str = titles["track1_OOF_stage_c"] + " PR-Curve"
except Exception:
    pass

plt.title(title_str, pad=15)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_file = os.path.join(FIG_DIR, "fig_pr_curves_updated.pdf")
plt.savefig(out_file)
print(f"Saved: {out_file}")
