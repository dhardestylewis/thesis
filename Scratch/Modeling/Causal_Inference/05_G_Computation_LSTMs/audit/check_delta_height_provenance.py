"""
check_delta_height_provenance.py
Trace delta_requested_height_ldc all the way from source to panel.

Chain:
  06_extract_zoning_codes.py
    -> model_ready_zoning_data.csv  [Requested_Zoning, Initial_Zoning]
  00a_engineer_model_ready_zoning.py
    -> model_ready_zoning_data.csv  [Requested_max_height_ft, Initial_max_height_ft]
  01_merge_panel.py (or equivalent)
    -> biweekly_panel.csv           [Requested_max_height_ft, Initial_max_height_ft]
  causal_cfm_cvae.py / causal_baselines.py
    -> delta_requested_height_ldc = clip(req_ht - init_ht, 0)
"""
import pandas as pd
import numpy as np

MRD   = r'c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv'
PANEL = r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv'

print("=== STEP 1: model_ready_zoning_data.csv ===")
mrd = pd.read_csv(MRD, low_memory=False)
for col in ['Requested_Zoning','Initial_Zoning','Final_Zoning',
            'Requested_max_height_ft','Initial_max_height_ft','Approved_max_height_ft']:
    s = mrd[col] if col in mrd.columns else None
    if s is None:
        print(f"  {col:<35} MISSING FROM FILE")
    else:
        nn = pd.to_numeric(s, errors='coerce').notna().sum() if 'height' in col else (s.fillna('').str.strip() != '').sum()
        print(f"  {col:<35} {nn:>5} / {len(mrd)} non-null")
print()

print("=== STEP 2: biweekly_panel.csv — which height cols are present? ===")
panel_cols = pd.read_csv(PANEL, low_memory=False, nrows=0).columns.tolist()
height_cols_in_panel = [c for c in panel_cols if 'height' in c.lower() or 'ht' in c.lower()]
print(f"  All height-related columns in panel ({len(height_cols_in_panel)}):")
for c in height_cols_in_panel:
    print(f"    {c}")
print()

# Key question: are the LDC lookup columns in the panel?
for col in ['Requested_max_height_ft', 'Initial_max_height_ft', 'Approved_max_height_ft']:
    present = col in panel_cols
    print(f"  {col:<35} {'IN PANEL' if present else 'NOT IN PANEL'}")
print()

# If present, check actual coverage
panel = pd.read_csv(PANEL, low_memory=False,
                    usecols=[c for c in ['case_number','Requested_max_height_ft',
                                         'Initial_max_height_ft'] if c in panel_cols])
if 'Requested_max_height_ft' in panel.columns:
    req = pd.to_numeric(panel['Requested_max_height_ft'], errors='coerce')
    init= pd.to_numeric(panel['Initial_max_height_ft'], errors='coerce') if 'Initial_max_height_ft' in panel.columns else pd.Series(dtype=float)
    delta = (req - init).clip(lower=0)
    n_cases = panel['case_number'].nunique()
    n_with_req = panel.loc[req.notna(), 'case_number'].nunique()
    n_with_both = panel.loc[req.notna() & init.notna(), 'case_number'].nunique()
    n_nonzero_delta = panel.loc[delta > 0, 'case_number'].nunique()
    print(f"  Panel cases total:              {n_cases}")
    print(f"  Cases with req_ht populated:    {n_with_req} ({n_with_req/n_cases:.1%})")
    print(f"  Cases with both req+init:       {n_with_both} ({n_with_both/n_cases:.1%})")
    print(f"  Cases with delta > 0 (signal):  {n_nonzero_delta} ({n_nonzero_delta/n_cases:.1%})")
    print(f"  => delta_requested_height_ldc would be nonzero for {n_nonzero_delta} cases")
else:
    print("  Requested_max_height_ft NOT IN PANEL — delta will be 0 everywhere")
    print()
    print("=== FINDING: Which pipeline script merges MRD cols into panel? ===")
    import os, glob
    pipeline_dir = r'c:\Users\dhl\data\Thesis\thesis\Scripts\pipeline'
    for fn in sorted(os.listdir(pipeline_dir)):
        if fn.endswith('.py'):
            fpath = os.path.join(pipeline_dir, fn)
            with open(fpath, encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            if 'Requested_max_height' in txt or 'model_ready_zoning' in txt:
                print(f"  {fn}: references height/zoning merge")
            elif 'biweekly_panel' in txt or 'panel' in txt.lower():
                print(f"  {fn}: panel-related (no height merge)")
