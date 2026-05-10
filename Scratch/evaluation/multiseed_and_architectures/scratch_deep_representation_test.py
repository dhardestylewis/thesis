import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

# --- PATHWAY A: NAIVE BASELINE (Exact replication of Gauntlet) ---
drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
df_naive = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
X_naive = df_naive.values

# --- PATHWAY B: PERFECT REPRESENTATION (KNN + Categorical Embeddings) ---
# Extract Numerics (with NaN preserved)
df_perf_num = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
num_cols = df_perf_num.columns.tolist()

# Run KNN Imputation (Limit to 5 neighbors)
print("[*] Running KNN Imputation on numerical gradient space...")
knn = KNNImputer(n_neighbors=5, weights='distance')
X_perf_num = knn.fit_transform(df_perf_num.values)

# Extract Categoricals
df_perf_cat = df[['council_district']].copy()
le = LabelEncoder()
df_perf_cat['council_district_enc'] = le.fit_transform(df_perf_cat['council_district'])
X_perf_cat = df_perf_cat[['council_district_enc']].values

# Combine
X_perf = np.column_stack([X_perf_num, X_perf_cat])
cat_idxs = [X_perf_num.shape[1]] # Explicitly use shape size to find index
print(f"X_perf_num bounds: {X_perf_num.shape}, placing cat at: {cat_idxs[0]}")
cat_dims = [len(le.classes_)]
cat_emb_dim = [4]

# --- SHARED EXPERIMENTAL SETUP ---
y = df['is_protested'].values
years = df['year'].values
anchor = 2020
train_mask = years < anchor

# Pre-2020 Splits
X_naive_train, X_naive_test_matrix = X_naive[train_mask], X_naive[~train_mask]
X_perf_train, X_perf_test_matrix = X_perf[train_mask], X_perf[~train_mask]
y_train = y[train_mask]

# Scaling constraints
sc_naive = StandardScaler()
X_naive_train_sc = sc_naive.fit_transform(X_naive_train)

sc_perf = StandardScaler()
X_perf_train_sc_num = sc_perf.fit_transform(X_perf_train[:, :-1]) # Don't scale categorical
X_perf_train_sc = np.column_stack([X_perf_train_sc_num, X_perf_train[:, -1]])

# MODELS
models = {
    'CatBoost_Benchmark': CatBoostClassifier(iterations=300, depth=6, random_seed=42, verbose=0),
    'TabNet_NaiveZeroes': TabNetClassifier(verbose=0, seed=42)
}

print("\n[*] Training Naive Architectures...")
fitted = {}
for name, m in models.items():
    if 'TabNet' in name:
        m.fit(X_naive_train_sc, y_train, max_epochs=20, patience=5)
    else:
        m.fit(X_naive_train, y_train)
    fitted[name] = m

print("[*] Training PERFECT TabNet with Categorical Embeddings and KNN Geometry...")
m_perf = TabNetClassifier(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=cat_emb_dim,
    verbose=0, 
    seed=42
)
m_perf.fit(X_perf_train_sc, y_train, max_epochs=20, patience=5)
fitted['TabNet_PerfectRepresentation'] = m_perf

# EVALUATION
eval_years = [2021, 2022, 2023, 2024]
results = []
print("[*] Evaluating Out-of-Sample Drift...")
for test_year in eval_years:
    mask = years == test_year
    y_test = y[mask]
    
    # Naive Setup
    X_naive_raw = X_naive[mask]
    X_naive_sc = sc_naive.transform(X_naive_raw)
    
    # Perf Setup
    X_perf_raw = X_perf[mask]
    X_perf_sc_num = sc_perf.transform(X_perf_raw[:, :-1])
    X_perf_sc = np.column_stack([X_perf_sc_num, X_perf_raw[:, -1]])
    
    for name, m in fitted.items():
        if name == 'TabNet_PerfectRepresentation':
            p = m.predict_proba(X_perf_sc)[:, 1]
        elif name == 'TabNet_NaiveZeroes':
            p = m.predict_proba(X_naive_sc)[:, 1]
        else:
            p = m.predict_proba(X_naive_raw)[:, 1]
            
        prauc = average_precision_score(y_test, p)
        results.append({
            'Model': name,
            'Evaluate_Year': test_year,
            'PRAUC': round(prauc, 3)
        })

res_df = pd.DataFrame(results)
print("\n=== Data Representation Geometry Test ===")
pivot = res_df.pivot_table(index='Model', columns='Evaluate_Year', values='PRAUC')
print(pivot.to_markdown())

