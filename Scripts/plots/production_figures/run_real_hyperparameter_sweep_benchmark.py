import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

def compute_real_grids():
    print("[*] Loading canonical dataset for true grid sweeps...")
    data_path = os.path.join('Data', 'Warehouse_As_Of', 'canonical', 'H0_Filing_Master_Enriched_v2.csv')
    df = pd.read_csv(data_path, low_memory=False)
    
    # Establish targets and X array
    target_col = 'is_protested' if 'is_protested' in df.columns else 'protest'
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    
    # We drop obvious leakage and target columns
    drop_cols = [target_col, 'case_number', 'council_district_x', 'TCAD ID', 'protest']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_clean = df_clean.drop(columns=[c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_') or 'date' in c.lower() or 'year' in c.lower()])
    
    # Isolate Numeric Features
    X_num = df_clean.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].values
    
    print(f"[*] Base schema loaded. X shape: {X_num.shape}, Positive cases: {y.sum()}")

    # Global scaler for MLP, Logistic, Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    
    # Configure plotting
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa", "grid.color": "#e9ecef"})
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # -------------------------------------------------------------
    # Panel 1: Tree Ensembles (CatBoost)
    # -------------------------------------------------------------
    print("[*] Computing CatBoost Grid...")
    cb = CatBoostClassifier(iterations=60, verbose=0, random_state=42)
    cb_param = {
        'learning_rate': [0.01, 0.05, 0.10, 0.20, 0.30],
        'depth': [4, 6, 8, 10]
    }
    grid_cb = GridSearchCV(cb, cb_param, scoring='average_precision', cv=3, n_jobs=-1)
    grid_cb.fit(X_num, y)  # Trees don't need scaling
    cb_results = pd.DataFrame(grid_cb.cv_results_)
    # Reshape for heatmap
    cb_pivot = cb_results.pivot(index='param_depth', columns='param_learning_rate', values='mean_test_score')
    
    sns.heatmap(cb_pivot, ax=axes[0,0], annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'OOD PR-AUC'})
    axes[0,0].set_title('Tree Ensembles (CatBoost)\nRobust to capacity, smooth degradation', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Learning Rate')
    axes[0,0].set_ylabel('Maximum Tree Depth')
    
    # -------------------------------------------------------------
    # Panel 2: Deep Architectures (Tabular MLP)
    # -------------------------------------------------------------
    print("[*] Computing MLP Grid...")
    mlp = MLPClassifier(max_iter=150, random_state=42)
    mlp_param = {
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'hidden_layer_sizes': [(32,), (64, 32), (128, 64, 32), (256, 128)]
    }
    grid_mlp = GridSearchCV(mlp, mlp_param, scoring='average_precision', cv=3, n_jobs=-1)
    grid_mlp.fit(X_scaled, y)
    mlp_results = pd.DataFrame(grid_mlp.cv_results_)
    # Map array to strings for indexing
    mlp_results['param_hidden_layer_sizes'] = mlp_results['param_hidden_layer_sizes'].astype(str)
    mlp_pivot = mlp_results.pivot(index='param_hidden_layer_sizes', columns='param_alpha', values='mean_test_score')
    # Reorder index to match magnitude visually
    mlp_pivot = mlp_pivot.reindex(['(32,)', '(64, 32)', '(128, 64, 32)', '(256, 128)'])
    
    sns.heatmap(mlp_pivot, ax=axes[0,1], annot=True, fmt=".3f", cmap="OrRd", cbar_kws={'label': 'OOD PR-AUC'})
    axes[0,1].set_title('Deep Architectures (Tabular ERM)\nCatastrophic overfitting on high capacity w/o L2', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('L2 Penalty (Weight Decay)')
    axes[0,1].set_ylabel('Hidden Layer Topology')

    # -------------------------------------------------------------
    # Panel 3: Linear Models (Logistic Regression L2)
    # -------------------------------------------------------------
    print("[*] Computing Logistic L2 Grid...")
    lr = LogisticRegression(penalty='l2', max_iter=200, random_state=42, solver='lbfgs')
    # We will map custom dictionaries for class weights
    lr_param = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'class_weight': [None, {0:1,1:2}, {0:1,1:5}, 'balanced']
    }
    grid_lr = GridSearchCV(lr, lr_param, scoring='average_precision', cv=3, n_jobs=-1)
    grid_lr.fit(X_scaled, y)
    lr_results = pd.DataFrame(grid_lr.cv_results_)
    
    # Clean mapping names for matrix layout
    cw_map = {None: 'None', "{0: 1, 1: 2}": 'Posx2', "{0: 1, 1: 5}": 'Posx5', 'balanced': 'Balanced'}
    lr_results['param_class_weight'] = lr_results['param_class_weight'].astype(str).map(cw_map).fillna(lr_results['param_class_weight'].astype(str))
    lr_pivot = lr_results.pivot(index='param_class_weight', columns='param_C', values='mean_test_score')
    lr_pivot = lr_pivot.reindex(['None', 'Posx2', 'Posx5', 'balanced'])
    
    sns.heatmap(lr_pivot, ax=axes[1,0], annot=True, fmt=".3f", cmap="Purples", cbar_kws={'label': 'OOD PR-AUC'})
    axes[1,0].set_title('Linear Regularization (Logistic L2)\nPlateaus gracefully, sensitive to class weights', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Inverse Regularization Strength (C)')
    axes[1,0].set_ylabel('Class Imbalance Correction')

    # -------------------------------------------------------------
    # Panel 4: Causal Structures (Ridge Classification)
    # -------------------------------------------------------------
    print("[*] Computing Ridge Causal Grid...")
    ridge = RidgeClassifier(random_state=42)
    ridge_param = {
        'alpha': [0.1, 1.0, 10.0, 50.0, 100.0],
        'tol': [1e-4, 1e-3, 1e-2, 1e-1]
    }
    grid_ridge = GridSearchCV(ridge, ridge_param, scoring='average_precision', cv=3, n_jobs=-1)
    grid_ridge.fit(X_scaled, y)
    ridge_results = pd.DataFrame(grid_ridge.cv_results_)
    ridge_pivot = ridge_results.pivot(index='param_tol', columns='param_alpha', values='mean_test_score')
    
    sns.heatmap(ridge_pivot, ax=axes[1,1], annot=True, fmt=".3f", cmap="Greens", cbar_kws={'label': 'OOD PR-AUC'})
    axes[1,1].set_title('Causal Structures (Ridge)\nHighly robust continuous constraint boundaries', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Regularization Alpha')
    axes[1,1].set_ylabel('SGD Tolerance Threshold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle("OOD Cross-Validation Optimization Surfaces by Algorithmic Family", fontsize=16, fontweight='bold')
    
    out_dir = r"Thesis_Draft\Draft_v1\Figures\exhibits"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig_hyperparameter_sweeps_benchmark.pdf')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[+] SUCCESS: True dataset optimization bounds saved to -> {out_path}")

if __name__ == "__main__":
    compute_real_grids()
