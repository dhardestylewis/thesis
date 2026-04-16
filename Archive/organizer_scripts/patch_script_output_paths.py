import os

THESIS = r'C:\Users\dhl\data\Thesis\thesis'

patches = {
    # (relative_path, old_string, new_string)

    r'scratch\data_build\scratch_rebuild_family.py': [
        (
            r"ROOT = r'C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Tables'",
            r"ROOT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1\Tables'",
        ),
    ],

    r'scripts\audit\tag_generated_tables.py': [
        (r"Thesis_Draft\Draft_v1\Tables\multi_horizon_results.tex",
         r"Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_09_multi_horizon_results.tex"),
        (r"Thesis_Draft\Draft_v1\Tables\calibration_benchmark.tex",
         r"Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_03_calibration_benchmark.tex"),
        (r"Thesis_Draft\Draft_v1\Tables\alternative_architectures.tex",
         r"Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_02_alternative_architectures.tex"),
    ],

    r'scripts\diagnostics\compute_ace.py': [
        (r'Thesis_Draft\Draft_v1\Tables\metrics_config.tex',
         r'Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_08_metrics_config.tex'),
        (r'Thesis_Draft/Draft_v1/Tables/metrics_config.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter4_performance/tbl_ch4_08_metrics_config.tex'),
    ],

    r'scripts\manuscript\generate_unclustered_table.py': [
        (r'Thesis_Draft/Draft_v1/Tables/unclustered_stability.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter4_performance/tbl_ch4_18_unclustered_stability.tex'),
        (r'Thesis_Draft\Draft_v1\Tables\unclustered_stability.tex',
         r'Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_18_unclustered_stability.tex'),
    ],

    r'scripts\pipeline\12_generate_extracted_tables.py': [
        (r'Thesis_Draft/Draft_v1/Tables/stagea_limits.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter4_performance/tbl_ch4_13_stagea_limits.tex'),
        (r'Thesis_Draft/Draft_v1/Tables/geographic_causal.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter6_causal/tbl_ch6_01_geographic_causal.tex'),
        (r'Thesis_Draft/Draft_v1/Tables/stage_b.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter4_performance/tbl_ch4_12_stage_b.tex'),
        (r'Thesis_Draft/Draft_v1/Tables/seed_summary.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter4_performance/tbl_ch4_11_seed_summary.tex'),
        (r'Thesis_Draft/Draft_v1/Tables/disqualification_matrix.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter4_performance/tbl_ch4_06_disqualification_matrix.tex'),
        (r'Thesis_Draft/Draft_v1/Tables/spuriousness_index.tex',
         r'Thesis_Draft/Draft_v1/Tables/chapter5_attribution/tbl_ch5_05_spuriousness_index.tex'),
    ],

    r'scripts\plots\generate_buffer_map.py': [
        (r'Thesis_Draft\Draft_v1\Figures\waller_buffer_map.png',
         r'Thesis_Draft\Draft_v1\Figures\ch2\fig_ch2_01_waller_buffer_map.png'),
        (r'Thesis_Draft/Draft_v1/Figures/waller_buffer_map.png',
         r'Thesis_Draft/Draft_v1/Figures/ch2/fig_ch2_01_waller_buffer_map.png'),
    ],

    r'scripts\plots\generate_real_buffer_map.py': [
        (r'Thesis_Draft\Draft_v1\Figures\waller_buffer_map.png',
         r'Thesis_Draft\Draft_v1\Figures\ch2\fig_ch2_01_waller_buffer_map.png'),
        (r'Thesis_Draft/Draft_v1/Figures/waller_buffer_map.png',
         r'Thesis_Draft/Draft_v1/Figures/ch2/fig_ch2_01_waller_buffer_map.png'),
    ],
}

for rel_path, subs in patches.items():
    fp = os.path.join(THESIS, rel_path)
    if not os.path.exists(fp):
        print('SKIP (not found): ' + rel_path)
        continue
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    new_text = text
    for old, new in subs:
        new_text = new_text.replace(old, new)
    if new_text != text:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print('Patched: ' + rel_path)
    else:
        print('No match: ' + rel_path)

print('\nDone.')
