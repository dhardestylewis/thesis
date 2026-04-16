import os, shutil

def mv(src, dst):
    if not os.path.exists(src):
        print('SKIP: ' + os.path.basename(src))
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(os.path.basename(src) + ' -> ' + os.path.relpath(os.path.dirname(dst), os.path.dirname(src)))

# ── Archive ───────────────────────────────────────────────────────────────────
ARC = r'C:\Users\dhl\data\Thesis\thesis\Archive'
os.makedirs(os.path.join(ARC, 'organizer_scripts'), exist_ok=True)
os.makedirs(os.path.join(ARC, 'misc'),              exist_ok=True)

# Move the organizer scripts we've been generating into their own subdir
for f in ['organize_batch2.py', 'organize_data.py', 'organize_directory.py']:
    mv(os.path.join(ARC, f), os.path.join(ARC, 'organizer_scripts', f))

# Loose misc files
mv(os.path.join(ARC, 'Slack_Update_Draft.md'), os.path.join(ARC, 'misc', 'Slack_Update_Draft.md'))
mv(os.path.join(ARC, 'cols.txt'),              os.path.join(ARC, 'misc', 'cols.txt'))

print('Archive:', sorted(os.listdir(ARC)))

# ── Top-level scripts/ ────────────────────────────────────────────────────────
SCR = r'C:\Users\dhl\data\Thesis\thesis\scripts'
os.makedirs(os.path.join(SCR, 'pipeline'),    exist_ok=True)
os.makedirs(os.path.join(SCR, 'audit'),       exist_ok=True)
os.makedirs(os.path.join(SCR, 'diagnostics'), exist_ok=True)
os.makedirs(os.path.join(SCR, 'manuscript'),  exist_ok=True)
os.makedirs(os.path.join(SCR, 'plots'),       exist_ok=True)
os.makedirs(os.path.join(SCR, 'archive'),     exist_ok=True)

# Numbered pipeline steps (00-13) + build script
pipeline = [
    '00_build_case_universe.py', '01_build_labels.py', '02_build_features.py',
    '03_build_splits.py', '04_train_stage_a.py', '05_train_stage_c.py',
    '06_calibrate_stage_c.py', '07_evaluate_stage_c.py', '08_run_meta_attribution.py',
    '08b_run_ablation_suite.py', '08c_run_did_causal.py', '09_run_audits.py',
    '10_export_manuscript_artifacts.py', '11_final_build_gate.py',
    '12_generate_extracted_tables.py', '13_render_prose_figures.py',
    'build_thesis.ps1',
]
for f in pipeline:
    mv(os.path.join(SCR, f), os.path.join(SCR, 'pipeline', f))

# Audit & verification tools
audit = [
    'audit_indices.py', 'check_dates.py', 'check_overlap.py', 'check_overlap2.py',
    'check_sizes.py', 'language_audit.py', 'lint_thesis_terminology.py',
    'manuscript_freshness_audit.py', 'track_tex_recency.py', 'verify_tex_assets.py',
    'read_metrics.py', 'tag_generated_tables.py',
]
for f in audit:
    mv(os.path.join(SCR, f), os.path.join(SCR, 'audit', f))

# Diagnostics / debug
diagnostics = [
    'check_draft_vs_comments.py', 'check_metrics.py', 'count_cases.py',
    'debug_tcad.py', 'diag.py', 'compute_ace.py', 'dump_did_table.py',
    'extract_comments.py', 'extract_comments_deep.py',
]
for f in diagnostics:
    mv(os.path.join(SCR, f), os.path.join(SCR, 'diagnostics', f))

# Manuscript / table patching
manuscript = [
    'generate_unclustered_table.py', 'update_tables.py', 'patch_features.py',
    'patch_orchestrator.py', 'revert_features.py', 'tcad_normalize.py',
    'run_visualizations.py',
]
for f in manuscript:
    mv(os.path.join(SCR, f), os.path.join(SCR, 'manuscript', f))

# Plot generators
plots = [
    'generate_buffer_map.py', 'generate_real_buffer_map.py',
    'plot_clustered_dynamic.py', 'plot_unclustered_dynamic.py',
]
for f in plots:
    mv(os.path.join(SCR, f), os.path.join(SCR, 'plots', f))

# Old / superceded
archive = ['old_causal.py', 'old_causal_utf8.py']
for f in archive:
    mv(os.path.join(SCR, f), os.path.join(SCR, 'archive', f))

# Catch anything remaining
for fname in sorted(os.listdir(SCR)):
    full = os.path.join(SCR, fname)
    if os.path.isfile(full):
        mv(full, os.path.join(SCR, 'diagnostics', fname))

print('scripts/:', sorted(os.listdir(SCR)))
