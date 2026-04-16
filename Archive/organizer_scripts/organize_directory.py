"""
Thesis top-level directory organizer.
Moves all loose files into a clean folder structure without touching
existing organised subdirectories or core pipeline entry-points.
"""
import os
import shutil

ROOT = r'C:\Users\dhl\data\Thesis\thesis'

# ── directories to create ────────────────────────────────────────────────────
DIRS = [
    'scratch/attribution',
    'scratch/evaluation',
    'scratch/data_build',
    'scratch/plots',
    'scratch/diagnostics',
    'scratch/misc',
    'Outputs/latex',
]

for d in DIRS:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def mv(src_rel, dst_rel):
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(ROOT, dst_rel)
    if not os.path.exists(src):
        print(f'  SKIP (not found): {src_rel}')
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f'  {src_rel}  ->  {dst_rel}')

# ── LaTeX compile artifacts ──────────────────────────────────────────────────
for ext in ['.aux', '.lof', '.log', '.lot', '.out', '.toc']:
    mv(f'Austin_NIMBY_Thesis_Draft{ext}', f'Outputs/latex/Austin_NIMBY_Thesis_Draft{ext}')
mv('texput.log', 'Outputs/latex/texput.log')

# ── Loose text / CSV outputs ─────────────────────────────────────────────────
for f in [
    'grid_tuning_results.csv',
    'grid_tuning_results_expanded.csv',
    'grid_tuning_results_full.csv',
    'grid_results_markdown.txt',
    'grid_results_universal_markdown.txt',
    'jmlr_pages.txt',
    'jmlr_pages_utf8.txt',
    'figures_audit.txt',
    'thesis_full_text.txt',
    'thesis_tex.txt',
    'tex_files.txt',
    'top_tex.txt',
    'RECENCY_AUDIT_2026.md',
]:
    mv(f, f'Outputs/{f}')

# ── scratch/plots ────────────────────────────────────────────────────────────
plots = [
    'scratch_expanded_plot.py',
    'scratch_replot_multiseed.py',
    'scratch_build_clustermap_gif.py',
    'scratch_plot_drift.py',
    'scratch_plot_final_results.py',
    'scratch_plot_megamatrix.py',
    'scratch_plot_megastack.py',
    'scratch_plot_offset.py',
    'scratch_plot_omnibus.py',
    'scratch_plot_omnibus_ext.py',
    'scratch_plot_omnibus_ltr.py',
    'scratch_plot_omnibus_nontree.py',
    'scratch_plot_performance.py',
    'scratch_plot_preclustered_lineplots.py',
    'scratch_plot_semantic_omnimap.py',
]
for f in plots:
    mv(f, f'scratch/plots/{f}')

# ── scratch/attribution ──────────────────────────────────────────────────────
attribution = [
    'scratch_ablation.py',
    'scratch_c40_ablation.py',
    'scratch_recursive_ablation.py',
    'scratch_attr_family.py',
    'scratch_attr_family_latex.py',
    'scratch_final_attr_v2.py',
    'scratch_ltr_attribution.py',
    'scratch_ltr_meta_attribution.py',
    'scratch_ltr_weighted_meta_attribution.py',
    'scratch_meta_attn.py',
    'scratch_meta_unified.py',
    'scratch_recursive_meta_attribution.py',
    'scratch_recursive_meta_omnilag.py',
    'scratch_preclustered_omnimap.py',
    'scratch_omnibus_clustermap.py',
    'scratch_omnibus_ext.py',
    'scratch_omnibus_ltr.py',
    'scratch_omnibus_nontree.py',
    'scratch_ood_domain_heatmap.py',
    'scratch_unsupervised_cluster_shap.py',
    'scratch_unsupervised_temporal_heatmap.py',
    'scratch_stacked_temporal_shap.py',
    'scratch_simple_cluster_shap.py',
    'scratch_extract_domain_dictionary.py',
    'scratch_cluster_features.py',
    'scratch_cluster_semantics.py',
    'scratch_catboost_absmax.py',
    'scratch_catboost_limit.py',
    'scratch_ltr_clustermap.py',
    'scratch_ltr_clustermap_sequential.py',
    'scratch_ltr_stats.py',
    'scratch_ltr_check.py',
    'scratch_reverse_metastack.py',
    'scratch_cascade_stacking.py',
    'scratch_dump_megamatrix.py',
    'scratch_ultimate_megamatrix.py',
    'scratch_ultimate_megastack.py',
    'scratch_universal_ltr_matrix.py',
]
for f in attribution:
    mv(f, f'scratch/attribution/{f}')

# ── scratch/evaluation ───────────────────────────────────────────────────────
evaluation = [
    'scratch_prauc_family.py',
    'scratch_prauc_family_deep.py',
    'scratch_prauc_family_latex.py',
    'scratch_prauc_family_latex4.py',
    'scratch_prauc_family_latex5.py',
    'scratch_prauc_lift.py',
    'scratch_prauc_lift2.py',
    'scratch_prauc_lift3.py',
    'scratch_multiseed_performance.py',
    'scratch_multiseed_prauc_table5.py',
    'scratch_multiseed_unclustered.py',
    'scratch_grid_search.py',
    'scratch_grid_agg.py',
    'scratch_grid_agg_full.py',
    'scratch_grid_plot.py',
    'scratch_horizon_regression.py',
    'scratch_nontree_regression.py',
    'scratch_deep_representation_test.py',
    'scratch_ext_ext_deep.py',
    'scratch_target_topography.py',
    'scratch_topography_expanded.py',
    'scratch_topography_ranking.py',
    'scratch_target_research.py',
    'scratch_topo_research.py',
    'scratch_winners.py',
    'scratch_master_7arch.py',
    'scratch_master_7arch_disc.py',
    'scratch_master_10arch_disc.py',
]
for f in evaluation:
    mv(f, f'scratch/evaluation/{f}')

# ── scratch/data_build ───────────────────────────────────────────────────────
data_build = [
    'scratch_rebuild_enriched.py',
    'scratch_rebuild_enriched_v2.py',
    'scratch_rebuild_family.py',
    'scratch_lag_engineering.py',
    'scratch_lag_everything.py',
    'scratch_multiparcel_agg.py',
    'scratch_spatial_parcel_join.py',
    'scratch_fixed_universal_geoagg.py',
    'scratch_universal_descriptive_geoagg.py',
    'scratch_descriptive_geoagg.py',
    'scratch_geoagg_breakdown.py',
    'scratch_geoagg_research.py',
    'scratch_geoagg_search.py',
    'scratch_geocode_geoids.py',
    'scratch_geocode_geoids_v2.py',
    'scratch_geocode_spatial_fallback.py',
    'scratch_symmetric_geoagg.py',
    'scratch_pull_acs.py',
    'scratch_download_ldb.py',
    'scratch_ethnicolr_v2.py',
    'scratch_fold_categoricals.py',
    'scratch_fd_bins.py',
    'scratch_inspect_bins.py',
    'scratch_inspect_floats.py',
    'scratch_id_formats.py',
    'scratch_patch.py',
]
for f in data_build:
    mv(f, f'scratch/data_build/{f}')

# ── scratch/diagnostics ──────────────────────────────────────────────────────
diagnostics = [
    'scratch_check_coverage.py',
    'scratch_check_geo.py',
    'scratch_check_meta.py',
    'scratch_check_target.py',
    'scratch_dep_check.py',
    'scratch_find_col.py',
    'scratch_find_coords.py',
    'scratch_inspect_rep.py',
    'scratch_key_diag.py',
    'scratch_key_diag2.py',
    'scratch_shift_bug.py',
    'scratch_compare.py',
    'scratch_test_imports.py',
]
for f in diagnostics:
    mv(f, f'scratch/diagnostics/{f}')

# ── anything remaining scratch_*.py → scratch/misc ──────────────────────────
for fname in sorted(os.listdir(ROOT)):
    if fname.startswith('scratch_') and fname.endswith('.py'):
        mv(fname, f'scratch/misc/{fname}')

print('\nDone. Organisation complete.')
