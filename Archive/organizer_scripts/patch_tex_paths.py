import os

DRAFT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1'

tex_files = [os.path.join(DRAFT, 'Austin_NIMBY_Thesis_Draft.tex')]
sections_dir = os.path.join(DRAFT, 'Sections')
for f in os.listdir(sections_dir):
    if f.endswith('.tex'):
        tex_files.append(os.path.join(sections_dir, f))

replacements = {
    # TABLES
    r'\input{Figures/Chapter4/fig_ipw_overlap_balance.tex}': r'\input{Tables/misc/fig_ipw_overlap_balance.tex}',
    r'\input{Tables/alternative_architectures.tex}': r'\input{Tables/chapter4_performance/alternative_architectures.tex}',
    r'\input{Tables/archetypal_attribution.tex}': r'\input{Tables/chapter5_attribution/archetypal_attribution.tex}',
    r'\input{Tables/archetypal_attribution_weighted.tex}': r'\input{Tables/chapter5_attribution/archetypal_attribution_weighted.tex}',
    r'\input{Tables/ipw_balance_diagnostics.tex}': r'\input{Tables/chapter6_causal/ipw_balance_diagnostics.tex}',
    r'\input{Tables/ipw_diagnostics_summary.tex}': r'\input{Tables/chapter6_causal/ipw_diagnostics_summary.tex}',
    r'\input{Tables/lib_ast.tex}': r'\input{Tables/chapter1_descriptive/lib_ast.tex}',
    r'\input{Tables/metrics_config.tex}': r'\input{Tables/chapter4_performance/metrics_config.tex}',
    r'\input{Tables/performance_integrity_audit.tex}': r'\input{Tables/chapter4_performance/performance_integrity_audit.tex}',
    r'\input{Tables/semantic_feature_mapping.tex}': r'\input{Tables/chapter5_attribution/semantic_feature_mapping.tex}',
    r'\input{Tables/summary_stats_table.tex}': r'\input{Tables/chapter1_descriptive/summary_stats_table.tex}',
    r'\input{Tables/temporal_drift_analysis.tex}': r'\input{Tables/chapter4_performance/temporal_drift_analysis.tex}',
    r'\input{Tables/temporal_drift_family.tex}': r'\input{Tables/chapter4_performance/temporal_drift_family.tex}',
    r'\input{Tables/temporal_drift_prauc_lift.tex}': r'\input{Tables/chapter4_performance/temporal_drift_prauc_lift.tex}',

    # FIGURES
    r'{Chapter4/F12_Opposition_PR.png}': r'{ch4/fig_ch4_20_opposition_pr_curve.png}',
    r'{Chapter4/F22_Joint_Policy_Map.png}': r'{ch4/fig_ch4_21_joint_policy_map.png}',
    r'{Chapter4/F23_Spatial_Error.png}': r'{ch4/fig_ch4_22_spatial_error.png}',
    r'{Chapter4/F8_Calibration.png}': r'{ch4/fig_ch4_23_calibration.png}',
    r'{Chapter4/StageA_Figure3_PR_Curves.png}': r'{ch4/fig_ch4_25_stagea_pr_curves.png}',
    r'{Chapter4/fig_all_stages_seed.pdf}': r'{ch4/fig_ch4_27_all_stages_seed.pdf}',
    r'{Chapter4/fig_attribution_longitudinal.pdf}': r'{ch4/fig_ch4_30_attribution_longitudinal.pdf}',
    r'{Chapter4/fig_fpr_fnr_longitudinal.pdf}': r'{ch4/fig_ch4_32_fpr_fnr_longitudinal.pdf}',
    r'{Chapter4/fig_ood_offset_decay.pdf}': r'{ch4/fig_ch4_34_ood_offset_decay.pdf}',
    r'{Chapter4/fig_ood_seed_variance.pdf}': r'{ch4/fig_ch4_35_ood_seed_variance.pdf}',
    r'{Chapter4/fig_seed_stability.pdf}': r'{ch4/fig_ch4_36_seed_stability.pdf}',
    r'{Chapter4/fig_stage_b_perclass_boxplot.pdf}': r'{ch4/fig_ch4_39_stage_b_perclass_boxplot.pdf}',
    r'{Chapter4/fig_stage_b_seed.pdf}': r'{ch4/fig_ch4_40_stage_b_seed.pdf}',

    r'{Chapter5/Electoral_Placebo_DiD.png}': r'{ch5/fig_ch5_30_electoral_placebo_did.png}',
    r'{Chapter5/F16_Petition_RD.png}': r'{ch5/fig_ch5_31_petition_rd.png}',
    r'{Chapter5/F17_DiD_EventStudy.png}': r'{ch5/fig_ch5_32_did_event_study.png}',
    r'{Chapter5/fig_causal_context_did.png}': r'{ch5/fig_ch5_34_causal_context_did.png}',

    r'{Exploratory_EDA/waller_buffer_map.png}': r'{ch2/fig_ch2_01_waller_buffer_map.png}',
    r'{Fig10_Hyperparameter_Sweeps.png}': r'{archive/Fig10_Hyperparameter_Sweeps.png}',
    
    r'{Figures/Archive_Pipelines/fig1_spatial_distribution.png}': r'{archive/fig1_spatial_distribution.png}',
    r'{Figures/Qualitative_Appendix/sample_protest_petition_v15.pdf}': r'{appendix/fig_app_04_sample_protest_petition_v15.pdf}',
    r'{Qualitative_Appendix/interview_guide.pdf}': r'{appendix/fig_app_01_interview_guide.pdf}',
    r'{Qualitative_Appendix/project_overview.pdf}': r'{appendix/fig_app_02_project_overview.pdf}',
    r'{Qualitative_Appendix/protocol_summary.pdf}': r'{appendix/fig_app_03_protocol_summary.pdf}',

    r'{SHAP_MetaClustering/meta_attribution_clustermap.pdf}': r'{ch5/fig_ch5_35_meta_attribution_clustermap.pdf}',
    r'{SHAP_MetaClustering/meta_attribution_clustermap_weighted.pdf}': r'{ch5/fig_ch5_36_meta_attribution_clustermap_weighted.pdf}',

    r'{Track1_Exhibits/fig_attribution_rank_stability_H0.pdf}': r'{exhibits/fig_attribution_rank_stability_H0.pdf}',
    r'{Track1_Exhibits/fig_calibration_ece_H0.pdf}': r'{exhibits/fig_calibration_ece_H0.pdf}',
    r'{Track1_Exhibits/fig_feature_importance_clustered_H0_Full.pdf}': r'{exhibits/fig_feature_importance_clustered_H0_Full.pdf}',
    r'{Track1_Exhibits/fig_policy_regimes_H3.pdf}': r'{exhibits/fig_policy_regimes_H3.pdf}',
    r'{Track1_Exhibits/fig_shap_beeswarm_H0_Full.pdf}': r'{exhibits/fig_shap_beeswarm_H0_Full.pdf}',
    r'{Track1_Exhibits/fig_temporal_drift.pdf}': r'{exhibits/fig_temporal_drift.pdf}',
    r'{Track1_Exhibits/fig_temporal_drift_H0.pdf}': r'{exhibits/fig_temporal_drift_H0.pdf}',
    r'{Track1_Exhibits/fig_temporal_drift_H3.pdf}': r'{exhibits/fig_temporal_drift_H3.pdf}',

    r'{Track1_Predictive/Figures/fig_clustered_dynamic.pdf}': r'{ch4/fig_ch4_50_clustered_dynamic.pdf}',
    r'{Track1_Predictive/Figures/fig_unclustered_dynamic.pdf}': r'{ch4/fig_ch4_51_unclustered_dynamic.pdf}',
}

total_changed = 0
for tf in tex_files:
    with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    modified = False
    new_text = text
    for old, new in replacements.items():
        if old in new_text:
            new_text = new_text.replace(old, new)
            modified = True
            
    if modified:
        with open(tf, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f"Patched: {os.path.basename(tf)}")
        total_changed += 1

print(f"\nDone. Patched {total_changed} file(s).")
