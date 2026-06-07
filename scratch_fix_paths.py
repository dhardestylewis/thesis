import os

root = r"c:\Users\dhl\data\Thesis\thesis\Thesis_Draft\GSAPP_Final_Submission"
files_to_check = ["Lewis_Daniel_GSAPPUP2026_Thesis.tex", "Lewis_2026_Defense.tex"]

replacements = {
    "archive/fig1_spatial_distribution.png": "archive/maps_and_diagrams/fig1_spatial_distribution.png",
    "exhibits/fig_pr_curves_multihorizon.pdf": "exhibits/calibration_and_pr/fig_pr_curves_multihorizon.pdf",
    "exhibits/fig_temporal_drift_prauc_testyear.pdf": "exhibits/temporal_drift/fig_temporal_drift_prauc_testyear.pdf",
    "exhibits/fig_temporal_drift_lift_testyear.pdf": "exhibits/temporal_drift/fig_temporal_drift_lift_testyear.pdf",
    "exhibits/fig_temporal_drift_prauc_offset.pdf": "exhibits/temporal_drift/fig_temporal_drift_prauc_offset.pdf",
    "exhibits/fig_temporal_drift_lift_offset.pdf": "exhibits/temporal_drift/fig_temporal_drift_lift_offset.pdf",
    "exhibits/fig_feature_importance_clustered_H0_Full.pdf": "exhibits/feature_importance/fig_feature_importance_clustered_H0_Full.pdf",
    "exhibits/fig_ch4_14_forecasting_interaction_shap.pdf": "exhibits/chapter_exhibits/fig_ch4_14_forecasting_interaction_shap.pdf",
    "exhibits/fig_ch4_14b_forecasting_interaction_grid.pdf": "exhibits/chapter_exhibits/fig_ch4_14b_forecasting_interaction_grid.pdf",
    "exhibits/fig_ch5_14_causal_dml_interaction_shap.pdf": "exhibits/chapter_exhibits/fig_ch5_14_causal_dml_interaction_shap.pdf",
    "exhibits/fig_ch5_14b_causal_dml_interaction_grid.pdf": "exhibits/chapter_exhibits/fig_ch5_14b_causal_dml_interaction_grid.pdf",
    "exhibits/mercatus_map.png": "exhibits/misc/mercatus_map.png",
    "exhibits/fig_pr_curves_updated.pdf": "exhibits/calibration_and_pr/fig_pr_curves_updated.pdf",
    "exhibits/fig_hyperparameter_sweeps_benchmark.pdf": "exhibits/misc/fig_hyperparameter_sweeps_benchmark.pdf"
}

for fname in files_to_check:
    fpath = os.path.join(root, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    modified = False
    for old_str, new_str in replacements.items():
        if old_str in content:
            content = content.replace(old_str, new_str)
            modified = True
            print(f"[{fname}] Replaced {old_str} -> {new_str}")
            
    if modified:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved {fname}")
