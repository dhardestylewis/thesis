import os
import re

file_path = r'C:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Visualization\Production_Figures\plot_Track1_exhibits_real.py'
with open(file_path, 'r') as f:
    text = f.read()

# 1. Update temporal drift baseline logic
new_drift = '''        # 2. Temporal Drift
        drift_file = str(AR.stage_c_drift(hz))
        if os.path.exists(drift_file):
            plt.figure(figsize=(7, 5))
            try:
                df_drift = pd.read_csv(drift_file)
                # Compute baselines dynamically
                df_gt = pd.read_csv(os.path.join(ROOT, "Data", "Warehouse_As_Of", f"{hz}_Filing_Master_Enriched.csv"), low_memory=False)
                df_gt['year'] = pd.to_numeric(df_gt['year'], errors='coerce')
                target = 'is_protested' if 'is_protested' in df_gt.columns else 'protest'
                df_gt[target] = pd.to_numeric(df_gt[target], errors='coerce').fillna(0).astype(int)
                
                if not df_drift.empty:
                    for anchor in df_drift['Anchor'].unique():
                        sub = df_drift[df_drift['Anchor'] == anchor]
                        plt.plot(sub['Offset'], sub['PR-AUC'], marker='o', label=f'Anchor < {anchor}')
                        # True baseline for this anchor is the mean of the training set
                        anchor_baseline = df_gt[df_gt['year'] < anchor][target].mean()
                        plt.axhline(anchor_baseline, color='gray', linestyle=':', alpha=0.5)

                    overall_baseline = df_gt[target].mean()
                    plt.axhline(y=overall_baseline, color='red', linestyle='--', alpha=0.7, label=f'Pooled Baseline ({overall_baseline:.3f})')
                    
                    plt.title(titles.get("stage_c_drift", "Temporal Drift").format(hz=hz_name), fontsize=14)
                    plt.xlabel('Years Out-of-Distribution (T + offset)', fontsize=12)
                    plt.ylabel('PR-AUC', fontsize=12)
                    plt.xticks([0, 1, 2, 3])
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True, alpha=0.3)
            except Exception as e:
                print('Drift plot error:', e)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_temporal_drift_{hz}.pdf"))
            print(f"  [+] Saved fig_temporal_drift_{hz}.pdf")'''

text = re.sub(r'# 2\. Temporal Drift.*?(?=# 3\. Policy)', new_drift + '\n\n', text, flags=re.DOTALL)

# 2. Update policy regimes baseline logic
new_regime = '''        # 3. Policy Regimes
        regimes_file = str(AR.stage_c_regimes(hz))
        if os.path.exists(regimes_file):
            plt.figure(figsize=(8, 5))
            try:
                df_reg = pd.read_csv(regimes_file)
                if not df_reg.empty:
                    bars = plt.bar(df_reg['Regime'], df_reg['PR-AUC'], color=['navy', 'orange', 'darkred'])
                    
                    # Custom baselines per bar
                    df_gt = pd.read_csv(os.path.join(ROOT, "Data", "Warehouse_As_Of", f"{hz}_Filing_Master_Enriched.csv"), low_memory=False)
                    df_gt['year'] = pd.to_numeric(df_gt['year'], errors='coerce')
                    target = 'is_protested' if 'is_protested' in df_gt.columns else 'protest'
                    df_gt[target] = pd.to_numeric(df_gt[target], errors='coerce').fillna(0).astype(int)
                    
                    # Pre-2022 Validation -> year < 2022
                    b1 = df_gt[df_gt['year'] < 2022][target].mean()
                    b2 = df_gt[df_gt['year'] == 2022][target].mean()
                    b3 = df_gt[df_gt['year'] >= 2023][target].mean()
                    baselines = [b1, b2, b3]
                    
                    for i, bar in enumerate(bars):
                        plt.hlines(baselines[i], bar.get_x(), bar.get_x() + bar.get_width(), color='red', linestyle='--', alpha=0.9, linewidth=2)
                        if i == 0:
                            plt.plot([], [], color='red', linestyle='--', label='True Slice Incidence (Baseline)')
                            
                    plt.legend(loc='upper right', fontsize=9)
                    plt.title(titles.get("stage_c_policy_regimes", "Policy Regimes").format(hz=hz_name), fontsize=14)
                    plt.ylabel('PR-AUC', fontsize=12)
                    plt.ylim(0, max(0.5, df_reg['PR-AUC'].max() * 1.2))
                    plt.grid(axis='y', alpha=0.3)
            except Exception as e:
                print('Regime baseline error:', e)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_policy_regimes_{hz}.pdf"))
            print(f"  [+] Saved fig_policy_regimes_{hz}.pdf")'''

text = re.sub(r'# 3\. Policy Regimes.*?(?=# 4b\. Clustered)', new_regime + '\n\n        ', text, flags=re.DOTALL)

# 3. Update SHAP and Importance calls
new_calls = '''        # 4. Temporal Iteration over Importance and SHAP
        for period in ['Full', 'Pre-2022', 'Post-2022']:
            _plot_clustered_importance(hz, hz_name, titles, period)
            _plot_shap_beeswarm(hz, hz_name, titles, period)'''
text = re.sub(r'# 4b\. Clustered.*?_plot_shap_beeswarm\(hz, hz_name, titles\)', new_calls, text, flags=re.DOTALL)

# 4. Update signatures of _plot functions
text = text.replace('def _plot_clustered_importance(hz, hz_name, titles):', 'def _plot_clustered_importance(hz, hz_name, titles, period="Full"):')
text = text.replace('def _plot_shap_beeswarm(hz, hz_name, titles):', 'def _plot_shap_beeswarm(hz, hz_name, titles, period="Full"):')

# 5. Insert slice logic into _plot_clustered_importance
slice_logic_imp = '''
        df_data['year'] = pd.to_numeric(df_data['year'], errors='coerce')
        if period == 'Pre-2022':
            df_data = df_data[df_data['year'] < 2022]
        elif period == 'Post-2022':
            df_data = df_data[df_data['year'] >= 2022]
        X_num = df_data.select_dtypes(include=[np.number])'''
text = re.sub(r'X_num = df_data\.select_dtypes\(include=\[np\.number\]\)', slice_logic_imp, text)

# Insert saving out suffix
text = text.replace('f"fig_feature_importance_clustered_{hz}.pdf"', 'f"fig_feature_importance_clustered_{hz}_{period}.pdf"')
text = text.replace('f"fig_shap_beeswarm_{hz}.pdf"', 'f"fig_shap_beeswarm_{hz}_{period}.pdf"')

# Insert slice logic into _plot_shap_beeswarm
slice_logic_shap = '''
        df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
        if period == 'Pre-2022':
            df_clean = df_clean[df_clean['year'] < 2022]
        elif period == 'Post-2022':
            df_clean = df_clean[df_clean['year'] >= 2022]
            
        drop_cols = '''
text = text.replace('        drop_cols = ', slice_logic_shap)

with open(file_path, 'w') as f:
    f.write(text)
print('Patched plot_Track1_exhibits_real.py successfully.')
