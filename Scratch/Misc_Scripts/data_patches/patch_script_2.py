import sys

filepath = r"c:\Users\dhl\data\Thesis\thesis\src\interpretation\drift_and_archetypes.py"
with open(filepath, "r") as f:
    text = f.read()

# 1. Fix Leakage
leak_fix = "    drop_cols = ['is_protested', 'case_number', 'reconstructed_petition_share', 'case_id', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x']"
text = text.replace("    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x']", leak_fix)

# 2. Fix Sorting
# Replace `for model in sorted(df['Model'].unique()):` with a custom order
sort_logic = """
        sort_order = ['CatBoost', 'XGBoost', 'Logistic (L2)', 'Spatial-FE Logistic', 'TabNet', 'Anchor Regression (Causal)']
        ordered_models = [m for m in sort_order if m in df['Model'].unique()] + [m for m in sorted(df['Model'].unique()) if m not in sort_order]
        for model in ordered_models:
"""
text = text.replace("        for model in sorted(df['Model'].unique()):", sort_logic)

with open(filepath, "w") as f:
    f.write(text)

print("Patch 2 applied successfully.")
