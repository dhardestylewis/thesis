"""
zone_divergence.py
How frequently does Requested_Zoning differ from Initial_Zoning?
And what does the height delta look like when they differ?
"""
import pandas as pd
import numpy as np
import re

mrd = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv', low_memory=False)
print(f"Total cases: {len(mrd)}")
print()

# Only rows where both are populated
both = mrd[
    (mrd['Requested_Zoning'].fillna('').str.strip() != '') &
    (mrd['Initial_Zoning'].fillna('').str.strip() != '')
].copy()
print(f"Cases with both Requested + Initial Zoning: {len(both)}")
print()

# Strip overlays to get base zone for comparison
def base_zone(z):
    if pd.isna(z) or not str(z).strip(): return None
    z = str(z).strip().upper()
    # Remove NP, CO, H, MU overlay suffixes
    z = re.sub(r'[-\s]?(NP|CO|MU|H|V|CURE|DBE|NCCD).*$', '', z)
    return z.strip()

both['req_base'] = both['Requested_Zoning'].apply(base_zone)
both['init_base'] = both['Initial_Zoning'].apply(base_zone)

same    = (both['req_base'] == both['init_base']).sum()
differ  = (both['req_base'] != both['init_base']).sum()
print(f"Requested == Initial (same base zone):  {same:>5} ({same/len(both):.1%})")
print(f"Requested != Initial (actual upzone):   {differ:>5} ({differ/len(both):.1%})")
print()

# Look at the height delta when they diverge
diverged = both[both['req_base'] != both['init_base']].copy()
req_ht  = pd.to_numeric(diverged['Requested_max_height_ft'], errors='coerce')
init_ht = pd.to_numeric(diverged['Initial_max_height_ft'], errors='coerce')
delta   = req_ht - init_ht

print(f"Among diverged cases ({len(diverged)}):")
print(f"  Both heights populated:       {(req_ht.notna() & init_ht.notna()).sum()}")
print(f"  req_ht > init_ht (upzone):    {(delta > 0).sum()} ({(delta>0).sum()/len(diverged):.1%})")
print(f"  req_ht == init_ht (no delta): {(delta == 0).sum()} ({(delta==0).sum()/len(diverged):.1%})")
print(f"  req_ht < init_ht (downzone):  {(delta < 0).sum()} ({(delta<0).sum()/len(diverged):.1%})")
print(f"  Mean height delta:            {delta.mean():.1f} ft")
print(f"  Median height delta:          {delta.median():.1f} ft")
print()

# Distribution of height deltas
print("Height delta distribution (requested - initial):")
print(delta.describe().to_string())
print()

# Most common zone transitions
diverged['transition'] = diverged['init_base'].fillna('?') + ' -> ' + diverged['req_base'].fillna('?')
print("Most common zone transitions (top 20):")
print(diverged['transition'].value_counts().head(20).to_string())
print()

# Key question: does the height delta from LDC lookup accurately reflect
# the ACTUAL policy request, or is it discretized into LDC table buckets?
# Check: how many unique height values exist in Requested_max_height_ft?
req_vals = pd.to_numeric(mrd['Requested_max_height_ft'], errors='coerce').dropna()
print(f"Unique values in Requested_max_height_ft: {req_vals.nunique()}")
print("Value counts (top 15):")
print(req_vals.value_counts().head(15).to_string())
print()
print("=> These are all LDC table values (35, 40, 60, 90, 120, 400 ft etc.)")
print("=> The 'height request' is entirely derived from zoning code lookup, not from")
print("   actual architect/developer proposals. It reflects zone-class max, not project max.")
print()

# How many of the 4920 panel cases have height data?
panel = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv',
                    low_memory=False, usecols=['case_number','pdf_requested_height_ft',
                                               'net_height_change','resolved'])
panel_cases = panel['case_number'].nunique()
panel_ht = pd.to_numeric(panel['pdf_requested_height_ft'], errors='coerce')
print(f"Panel cases: {panel_cases}")
print(f"Panel cases with pdf_requested_height_ft: "
      f"{panel.loc[panel_ht.notna(), 'case_number'].nunique()} "
      f"({panel.loc[panel_ht.notna(), 'case_number'].nunique()/panel_cases:.1%})")
