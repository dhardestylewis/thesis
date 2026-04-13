import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from artifact_registry import ROOT_DIR, DATA_WAREHOUSE_DIR, TraceabilityRegistry as AR

def run_stage_b():
    df = pd.read_csv(str(DATA_WAREHOUSE_DIR / 'H0_Filing_Master_Enriched.csv'), on_bad_lines='skip', engine='python')
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
    report = classification_report(y, preds, zero_division=0, output_dict=True)
    
    print(f"Macro-F1 (6-Tier): {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y, preds, zero_division=0))

    # --- CONTINUOUS REGRESSION FOR HUMAN-READABLE METRICS ---
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    df['implied_sqft'] = df['delta_max_far'] * df['gross_site_area_acres'] * 43560
    y_sqft = df['implied_sqft']
    
    seeds = [10, 42, 100, 314, 777, 1234, 2023, 2024, 8080, 9999]
    mae_sqft_list = []
    mae_units_list = []

    for seed in seeds:
        X_train_sqft, X_val_sqft, y_train_sqft, y_val_sqft = train_test_split(X, y_sqft, test_size=0.1, random_state=seed)
        cbr = CatBoostRegressor(iterations=60, depth=4, learning_rate=0.08, loss_function='MAE', random_seed=seed, verbose=0)
        cbr.fit(X_train_sqft, y_train_sqft, eval_set=(X_val_sqft, y_val_sqft), early_stopping_rounds=15)
        
        preds_sqft = cbr.predict(X)
        mae_sqft = mean_absolute_error(y_sqft, preds_sqft)
        mae_sqft_list.append(mae_sqft)
        mae_units_list.append(mae_sqft / 1000.0)
    
    mean_mae_sqft = np.mean(mae_sqft_list)
    std_mae_sqft = np.std(mae_sqft_list)
    mean_mae_units = np.mean(mae_units_list)
    std_mae_units = np.std(mae_units_list)

    print(f"\\n--- Continuous Scale Error (10 Seeds) ---")
    print(f"Regression MAE: {mean_mae_sqft:,.0f} +/- {std_mae_sqft:,.0f} sqft")
    print(f"Regression Equivalent Unit MAE: {mean_mae_units:.1f} +/- {std_mae_units:.1f} units\\n")

    try:
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=mae_units_list, color='lightblue', width=0.3)
        sns.swarmplot(y=mae_units_list, color='darkblue', size=8)
        plt.title('Stage B: Multi-Seed Continuous Regression MAE\\n(10 Random Seeds)')
        plt.ylabel('Mean Absolute Error (Equivalent Units)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = Path(str(AR.FIGURES_DIR).replace('Draft_v1\\\\Figures', 'Draft_v1/Figures')) / "Chapter4" / "fig_stage_b_continuous_seed_mae.pdf"
        os.makedirs(plot_path.parent, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Saved Stage B continuous seed plot to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"    [!] Failed to plot multi-seed continuous MAE: {e}")


    # --- ALTERNATIVE ARCHITECTURES BENCHMARKS ---
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import warnings
    warnings.filterwarnings('ignore')

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X).flatten()
    f1_rf = f1_score(y, preds_rf, average='macro', zero_division=0)
    print(f"Random Forest Macro-F1: {f1_rf:.4f}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X).flatten()
    f1_lr = f1_score(y, preds_lr, average='macro', zero_division=0)
    print(f"Logistic Regression Macro-F1: {f1_lr:.4f}")

    # LightGBM
    from lightgbm import LGBMClassifier
    lgb = LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    lgb.fit(X_train, y_train)
    preds_lgb = lgb.predict(X).flatten()
    f1_lgb = f1_score(y, preds_lgb, average='macro', zero_division=0)
    print(f"LightGBM Macro-F1: {f1_lgb:.4f}")

    report_rf = classification_report(y, preds_rf, zero_division=0, output_dict=True)
    report_lr = classification_report(y, preds_lr, zero_division=0, output_dict=True)
    report_lgb = classification_report(y, preds_lgb, zero_division=0, output_dict=True)

    try:
        import sys
        module_path = os.path.join(os.path.dirname(__file__), '..')
        if module_path not in sys.path:
            sys.path.append(module_path)
            
        from Utilities_and_Logs.lib_metrics import update_metric
        update_metric("metricStageBMacroFOne", f"{f1:.3f}")
        update_metric("metricStageBRfMacroFOne", f"{f1_rf:.3f}")
        update_metric("metricStageBLogMacroFOne", f"{f1_lr:.3f}")
        update_metric("metricStageBLgbMacroFOne", f"{f1_lgb:.3f}")
        update_metric("metricStageBMaeSqft", f"${mean_mae_sqft:,.0f} \\pm {std_mae_sqft:,.0f}$")
        update_metric("metricStageBMaeUnits", f"${mean_mae_units:.1f} \\pm {std_mae_units:.1f}$")

        for label, metric_name in [
            ("PUD / large negotiated project", "PUD"),
            ("discretionary rezoning", "Rezoning"),
            ("by-right infill", "ByRight"),
            ("missing-middle", "MissingMiddle"),
            ("mixed-use", "MixedUse"),
            ("multifamily", "Multifamily")
        ]:
            update_metric(f"metricStageBCb{metric_name}FOne", f"{report.get(label, {}).get('f1-score', 0):.3f}")
            update_metric(f"metricStageBRf{metric_name}FOne", f"{report_rf.get(label, {}).get('f1-score', 0):.3f}")
            update_metric(f"metricStageBLgb{metric_name}FOne", f"{report_lgb.get(label, {}).get('f1-score', 0):.3f}")
            update_metric(f"metricStageBLog{metric_name}FOne", f"{report_lr.get(label, {}).get('f1-score', 0):.3f}")
    except Exception as e:
        print(f"    [!] Macro Telemetry Export Failed: {e}")

    out_path = str(AR.STAGE_B_MODEL)
    cb.save_model(out_path)
    print(f"Saved Stage B model to {out_path}")

if __name__ == '__main__':
    run_stage_b()
