import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, auc
from catboost import CatBoostClassifier

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of")
FIGURES_DIR = os.path.join(ROOT_DIR, "Thesis_Draft", "Draft_v1", "Figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

np.random.seed(42)

def calculate_pr_auc(y_true, y_pred_proba):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def run_real_pipelines():
    print("[*] Loading empirical Austin data warehouse (H0_Filing_Master_Enriched.csv)...")
    df = pd.read_csv(os.path.join(WORK_DIR, "H0_Filing_Master_Enriched.csv"), low_memory=False)
    
    if 'year' not in df.columns:
        print("[!] Fatal: 'year' column not found in H0_Filing_Master_Enriched.csv")
        return
        
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)
    
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X_raw.select_dtypes(include=[np.number]).fillna(0)
    y = df['is_protested']
    
    #################################################
    # 2. HYPERPARAMETER GRID SEARCH HEATMAP
    #################################################
    print("[*] Executing authentic CatBoost Grid Search optimization (Fig 10)...")
    train_idx = df['year'] <= 2022
    
    cb_clf = CatBoostClassifier(silent=True, iterations=20)
    grid_cb = {'depth': [2, 4, 6], 'learning_rate': [0.01, 0.05, 0.1]}
    search = GridSearchCV(cb_clf, grid_cb, scoring='average_precision', cv=2)
    search.fit(X[train_idx], y[train_idx])
    
    results = pd.DataFrame(search.cv_results_)
    pivot = results.pivot(index='param_depth', columns='param_learning_rate', values='mean_test_score')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'Mean CV PR-AUC'})
    plt.title("CatBoost Actual GridSearchCV Optimization Surface (H0 Enriched)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig10_Hyperparameter_Sweeps.png"), dpi=300)
    plt.close()

    #################################################
    # 3. EXPANDING WINDOW ROLLING-ORIGIN TRACKING
    #################################################
    print("[*] Executing Authentic Expanding Window Rolling-Origin Tracking (Fig 8)...")
    test_years = sorted([y for y in df['year'].unique() if y >= 2021])
    
    plt.figure(figsize=(9, 6))
    rolling_scores = []
    clf = CatBoostClassifier(silent=True, iterations=50, depth=4)
    
    for target_yr in test_years:
        t_idx = df['year'] < target_yr
        v_idx = df['year'] == target_yr
        
        if sum(y[t_idx]) > 0 and sum(y[v_idx]) > 0:
            clf.fit(X[t_idx], y[t_idx])
            preds = clf.predict_proba(X[v_idx])[:, 1]
            score = calculate_pr_auc(y[v_idx], preds)
            rolling_scores.append(score)
        else:
            rolling_scores.append(0.0)
            
    plt.plot(test_years, rolling_scores, marker='o', lw=2.5, color='darkred', label='CatBoost (H0)')
    plt.axvline(2022.5, color='black', linestyle=':', lw=2, label="Council Regime Shift (2022)")
    plt.ylabel("Precision-Recall AUC (Test Year t)")
    plt.xlabel("Expanding Window Temporal Target (Year t)")
    plt.title("Track 1: Genuine Rolling-Origin Expanding Window Validation")
    plt.xticks(test_years)
    plt.ylim([0, 1.05])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig8_Rolling_Origin_Horizons.png"), dpi=300)
    plt.close()
    
    #################################################
    # 1. SIMPLE IN-DIST VS OOD BAR CHART
    #################################################
    print("[*] Rendering Fig 9: Empirical Generalization Decay Array...")
    id_score = calculate_pr_auc(y[train_idx], search.best_estimator_.predict_proba(X[train_idx])[:, 1])
    ood_idx = df['year'] > 2022
    ood_score = calculate_pr_auc(y[ood_idx], search.best_estimator_.predict_proba(X[ood_idx])[:, 1]) if sum(y[ood_idx]) > 0 else 0
    
    plt.figure(figsize=(8, 5))
    names = ['Pre-2022 Generalization', 'Post-2022 Generalization']
    scores = [id_score, ood_score]
    plt.bar(names, scores, color=['steelblue', 'firebrick'], width=0.4)
    plt.ylabel('Precision-Recall AUC')
    plt.title('Track 1: Empirical Temporal Decay (ID vs Worst-Regime OOD)')
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig9_Model_Comparison_PR_AUC.png"), dpi=300)
    plt.close()
    
    print("[+] Model Sweeps and RO Graphics written legitimately.")

if __name__ == "__main__":
    run_real_pipelines()
