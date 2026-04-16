"""
Comprehensive fix:
1. Patch internal data/output paths in all scripts
2. Rename scripts to match output slug scheme where appropriate
3. Update any references to renamed scripts
"""
import os, re, shutil

THESIS = r'C:\Users\dhl\data\Thesis\thesis'

def patch_file(rel_path, substitutions):
    """Apply string substitutions to a file in-place."""
    fp = os.path.join(THESIS, rel_path)
    if not os.path.exists(fp):
        print(f'SKIP (missing): {rel_path}')
        return
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    new_text = text
    for old, new in substitutions:
        new_text = new_text.replace(old, new)
    if new_text != text:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f'Patched: {rel_path}')
    else:
        print(f'No match: {rel_path}')

def rename_script(rel_src, rel_dst):
    """Rename a script file."""
    src = os.path.join(THESIS, rel_src)
    dst = os.path.join(THESIS, rel_dst)
    if not os.path.exists(src):
        print(f'SKIP (missing): {rel_src}')
        return rel_src, rel_dst
    if os.path.exists(dst):
        print(f'SKIP (exists): {rel_dst}')
        return rel_src, rel_dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f'Renamed: {os.path.basename(rel_src)} -> {os.path.basename(rel_dst)}')
    return rel_src, rel_dst

# ── Canonical data path constants (for patch substitutions) ──────────────────
WAREHOUSE_V1    = 'Data/Warehouse_As_Of/H0_Filing_Master_Enriched.csv'
WAREHOUSE_V1_BS = r'Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv'
WAREHOUSE_V2    = 'Data/Warehouse_As_Of/canonical/H0_Filing_Master_Enriched_v2.csv'
WAREHOUSE_V2_BS = r'Data\Warehouse_As_Of\canonical\H0_Filing_Master_Enriched_v2.csv'

PANEL_GEO_OLD  = 'Data/Panel/case_geoid_lookup.csv'
PANEL_GEO_NEW  = 'Data/Panel/geo/case_geoid_lookup.csv'
PANEL_PROP_OLD = 'Data/Panel/property_universe.csv'
PANEL_PROP_NEW = 'Data/Panel/parcel/property_universe.csv'

ABS_WH_V1_LC = r'c:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Complete.csv'
ABS_WH_V1_NEW = r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical\H0_Filing.csv'

ABS_PANEL_OLD = r'c:\Users\dhl\data\thesis\thesis\Data\Panel\Output\Property_Year_Panel_Enriched.csv'
ABS_PANEL_NEW = r'C:\Users\dhl\data\Thesis\thesis\Data\Panel\parcel\property_universe.csv'

# Old output locations for plots
PLOT_CLU_OLD = 'c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track1_Predictive/Figures/fig_clustered_dynamic'
PLOT_CLU_NEW = 'C:/Users/dhl/data/Thesis/thesis/Thesis_Draft/Draft_v1/Figures/ch4/fig_ch4_50_clustered_dynamic'
PLOT_UCL_OLD = 'c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track1_Predictive/Figures/fig_unclustered_dynamic'
PLOT_UCL_NEW = 'C:/Users/dhl/data/Thesis/thesis/Thesis_Draft/Draft_v1/Figures/ch4/fig_ch4_51_unclustered_dynamic'

ABS_WH_V1_ATT  = r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of'
ABS_WH_V1_ATT2 = r'c:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of'
ABS_WH_V2_ATT  = r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical'

DEAD_SCRIPT = r'c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Visualization\Production_Figures\plot_F17_DiD_real.py'
DEAD_SCRIPT_NEW = r'C:\Users\dhl\data\Thesis\thesis\scratch\plots\scratch_plot_did_event_study.py'

# ── 1. DATA PATH PATCHES ─────────────────────────────────────────────────────

patch_file(r'scripts\diagnostics\debug_tcad.py', [
    (ABS_WH_V1_LC, ABS_WH_V1_NEW),
    (ABS_PANEL_OLD, ABS_PANEL_NEW),
    (r'c:\Users\dhl\data\thesis\thesis', r'C:\Users\dhl\data\Thesis\thesis'),
])

patch_file(r'scripts\diagnostics\diag.py', [
    (WAREHOUSE_V1, WAREHOUSE_V2),
    (WAREHOUSE_V1_BS, WAREHOUSE_V2_BS),
])

patch_file(r'scripts\diagnostics\dump_did_table.py', [
    (r'c:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv',
     r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical\H0_Filing_Master_Enriched_v2.csv'),
])

patch_file(r'scripts\plots\plot_clustered_dynamic.py', [
    (PLOT_CLU_OLD + '.pdf', PLOT_CLU_NEW + '.pdf'),
    (PLOT_CLU_OLD + '.png', PLOT_CLU_NEW + '.png'),
])

patch_file(r'scripts\plots\plot_unclustered_dynamic.py', [
    (PLOT_UCL_OLD + '.pdf', PLOT_UCL_NEW + '.pdf'),
    (PLOT_UCL_OLD + '.png', PLOT_UCL_NEW + '.png'),
])

patch_file(r'Thesis_Draft\Draft_v1\scripts\patching\patch.py', [
    (DEAD_SCRIPT, DEAD_SCRIPT_NEW),
])

for rel in [
    r'scratch\attribution\scratch_c40_ablation.py',
    r'scratch\attribution\scratch_recursive_ablation.py',
]:
    patch_file(rel, [
        (ABS_WH_V1_ATT,  ABS_WH_V2_ATT),
        (ABS_WH_V1_ATT2, ABS_WH_V2_ATT),
    ])

for rel in [
    r'scratch\data_build\scratch_geocode_geoids_v2.py',
    r'scratch\data_build\scratch_geocode_spatial_fallback.py',
    r'scratch\diagnostics\scratch_check_geo.py',
]:
    patch_file(rel, [(PANEL_GEO_OLD, PANEL_GEO_NEW)])

for rel in [r'scratch\diagnostics\scratch_check_coverage.py']:
    patch_file(rel, [
        (PANEL_PROP_OLD, PANEL_PROP_NEW),
        (PANEL_GEO_OLD,  PANEL_GEO_NEW),
        ('Data/Panel/Intermediate', 'Data/Panel/Intermediate'),  # already fine
    ])

for rel in [
    r'scratch\data_build\scratch_id_formats.py',
    r'scratch\diagnostics\scratch_dep_check.py',
    r'scratch\diagnostics\scratch_key_diag.py',
    r'scratch\diagnostics\scratch_key_diag2.py',
]:
    patch_file(rel, [
        (WAREHOUSE_V1,  WAREHOUSE_V2),
        (WAREHOUSE_V1_BS, WAREHOUSE_V2_BS),
        ('Data/Warehouse_As_Of/', 'Data/Warehouse_As_Of/canonical/'),
    ])

# ── 2. RENAME GENERATOR SCRIPTS to match their output slugs ──────────────────
script_renames = [
    # plots/
    (r'scripts\plots\generate_buffer_map.py',
     r'scripts\plots\gen_fig_ch2_01_waller_buffer_map.py'),
    (r'scripts\plots\generate_real_buffer_map.py',
     r'scripts\plots\gen_fig_ch2_01_waller_buffer_map_real.py'),
    (r'scripts\plots\plot_clustered_dynamic.py',
     r'scripts\plots\gen_fig_ch4_50_clustered_dynamic.py'),
    (r'scripts\plots\plot_unclustered_dynamic.py',
     r'scripts\plots\gen_fig_ch4_51_unclustered_dynamic.py'),
    # manuscript/
    (r'scripts\manuscript\generate_unclustered_table.py',
     r'scripts\manuscript\gen_tbl_ch4_18_unclustered_stability.py'),
    # diagnostics/
    (r'scripts\diagnostics\compute_ace.py',
     r'scripts\diagnostics\gen_tbl_ch4_08_metrics_config.py'),
]

rename_map = {}
for old_rel, new_rel in script_renames:
    rename_script(old_rel, new_rel)
    rename_map[os.path.basename(old_rel)] = os.path.basename(new_rel)

# ── 3. PATCH ORCHESTRATORS that call renamed scripts ─────────────────────────
orchestrators = [
    r'scripts\pipeline\12_generate_extracted_tables.py',
    r'scripts\pipeline\13_render_prose_figures.py',
    r'scripts\pipeline\08_run_meta_attribution.py',
    r'scripts\pipeline\09_run_audits.py',
    r'scripts\pipeline\10_export_manuscript_artifacts.py',
    r'scripts\pipeline\11_final_build_gate.py',
    r'scripts\manuscript\patch_orchestrator.py',
    r'Thesis_Draft\Draft_v1\scripts\patching\patch.py',
]

for orch in orchestrators:
    subs = [(old_name, new_name) for old_name, new_name in rename_map.items()]
    if subs:
        patch_file(orch, subs)

print('\nAll done.')
