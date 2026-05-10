import sys
import os

filepath = r"c:\Users\dhl\data\Thesis\thesis\src\interpretation\drift_and_archetypes.py"
with open(filepath, "r") as f:
    text = f.read()

# 1. Modify function definition
text = text.replace("def run_drift_and_archetypes():", "def run_drift_and_archetypes(threshold=0.20, is_appendix=False):")

# 2. Modify ground truth data
data_injection = """
    df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2.csv'), low_memory=False)
    reg = pd.read_parquet(os.path.join(ROOT, 'registries', 'label_registry.parquet'))
    reg_v1 = reg[reg['label_version']=='label_v1_reconstructed_threshold_crossing']
    df = df.merge(reg_v1[['case_id', 'reconstructed_petition_share']], left_on='case_number', right_on='case_id', how='left')
    df['reconstructed_petition_share'] = df['reconstructed_petition_share'].fillna(0)
    df['is_protested'] = (df['reconstructed_petition_share'] > threshold).astype(int)
"""
text = text.replace("    df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2.csv'), low_memory=False)", data_injection)

# 3. Prevent figures from regenerating endlessly during appendices loop
text = text.replace("""    out_dir_fig = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "ch5")""", """
    if is_appendix: return  # Skip figures for appendices loop
    out_dir_fig = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "ch5")""")

# 4. Modify saving folder
tbl_injection = """
    OUT_DIR = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Tables')
    if is_appendix:
        perf_dir = os.path.join(OUT_DIR, 'appendices_drift')
        attr_dir = os.path.join(OUT_DIR, 'appendices_drift')
        os.makedirs(perf_dir, exist_ok=True)
        suffix = f"_t{int(threshold*100):02d}"
    else:
        perf_dir = os.path.join(OUT_DIR, 'chapter4_performance')
        attr_dir = os.path.join(OUT_DIR, 'chapter5_attribution')
        suffix = ""
    
"""
text = text.replace("    OUT_DIR = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Tables')", tbl_injection)

# 5. Fix os.path.join directories for the Tables
text = text.replace("os.path.join(OUT_DIR, 'chapter4_performance', 'tbl_ch4_14_temporal_drift_analysis.tex')", "os.path.join(perf_dir, f'tbl_ch4_14_temporal_drift_analysis{suffix}.tex')")
text = text.replace("os.path.join(OUT_DIR, 'chapter4_performance', 'tbl_ch4_17_temporal_drift_prauc_lift.tex')", "os.path.join(perf_dir, f'tbl_ch4_17_temporal_drift_prauc_lift{suffix}.tex')")
text = text.replace("os.path.join(OUT_DIR, 'chapter4_performance', 'tbl_ch4_15_temporal_drift_family.tex')", "os.path.join(perf_dir, f'tbl_ch4_15_temporal_drift_family{suffix}.tex')")

text = text.replace("os.path.join(OUT_DIR, 'chapter5_attribution', \"tbl_ch5_02_archetypal_attribution.tex\")", "os.path.join(attr_dir, f'tbl_ch5_02_archetypal_attribution{suffix}.tex')")
text = text.replace("os.path.join(OUT_DIR, 'chapter5_attribution', \"tbl_ch5_03_archetypal_attribution_weighted.tex\")", "os.path.join(attr_dir, f'tbl_ch5_03_archetypal_attribution_weighted{suffix}.tex')")

with open(filepath, "w") as f:
    f.write(text)

print("Patch applied to drift_and_archetypes.py.")
