"""
height_tier_crossings.py
Properly count how many zone transitions actually cross LDC height tiers.
Don't just count req != init — count how many cross the 35/40/60/90/120/400 boundary.
"""
import pandas as pd
import numpy as np
import re

LDC_HT = {
    'RR':35,'LA':35,'DR':35,'SF-1':35,'SF-2':35,'SF-3':35,'SF-4A':35,'SF-4B':35,
    'SF-5':35,'SF-6':35,'SF':35,'MH':35,'MF-1':40,'MF-2':40,'MF-3':40,'MF-4':60,
    'MF-5':60,'MF-6':90,'MF':60,'NO':35,'LO':40,'GO':60,'CR':35,'LR':40,'GR':60,
    'CS':60,'CS-1':60,'CH':120,'IP':60,'LI':60,'MI':60,'HI':60,'CBD':400,'DMU':120,
    'TOD':60,'PUD':60,'ERC':60,'P':60,'AG':35,'W':35
}

def base_zone(z):
    if pd.isna(z) or not str(z).strip(): return None
    z = str(z).strip().upper()
    z = re.sub(r'[-\s]?(NP|CO|MU|H|V|CURE|DBE|NCCD).*$', '', z)
    return z.strip()

def ldc_ht(z):
    b = base_zone(z)
    if b and b in LDC_HT: return LDC_HT[b]
    if b:
        b2 = b.split('-')[0]
        if b2 in LDC_HT: return LDC_HT[b2]
    return None

mrd = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv', low_memory=False)

both = mrd[
    (mrd['Requested_Zoning'].fillna('').str.strip() != '') &
    (mrd['Initial_Zoning'].fillna('').str.strip() != '')
].copy()

both['req_ht_ldc']  = both['Requested_Zoning'].apply(ldc_ht)
both['init_ht_ldc'] = both['Initial_Zoning'].apply(ldc_ht)
both['delta_ldc']   = both.apply(
    lambda r: r['req_ht_ldc'] - r['init_ht_ldc']
    if pd.notna(r['req_ht_ldc']) and pd.notna(r['init_ht_ldc']) else np.nan, axis=1)

both_valid = both.dropna(subset=['delta_ldc'])
print(f"Cases with both zones LDC-mapped: {len(both_valid)} / {len(both)}")
print()

n_cross_up   = (both_valid['delta_ldc'] > 0).sum()
n_same_tier  = (both_valid['delta_ldc'] == 0).sum()
n_cross_down = (both_valid['delta_ldc'] < 0).sum()
total = len(both_valid)

print("=== HEIGHT TIER CROSSING COUNTS (from LDC lookup) ===")
print(f"  req_ht > init_ht (height tier UP):   {n_cross_up:>5}  ({n_cross_up/total:.1%})")
print(f"  req_ht == init_ht (same tier):        {n_same_tier:>5}  ({n_same_tier/total:.1%})")
print(f"  req_ht < init_ht (tier DOWN):         {n_cross_down:>5}  ({n_cross_down/total:.1%})")
print()

# Tier upward crossings by magnitude
up = both_valid[both_valid['delta_ldc'] > 0].copy()
print(f"Height tier UPWARD crossings: {len(up)}")
print("Tier jump distribution:")
print(up['delta_ldc'].value_counts().sort_index().to_string())
print()

print("Most common upzone transitions that cross height tier:")
up['transition'] = (up['Initial_Zoning'].apply(base_zone).fillna('?') + ' (' +
                    up['init_ht_ldc'].astype(str) + 'ft) -> ' +
                    up['Requested_Zoning'].apply(base_zone).fillna('?') + ' (' +
                    up['req_ht_ldc'].astype(str) + 'ft)')
print(up['transition'].value_counts().head(25).to_string())
print()

# Now specifically for Approved cases — what is the scale of the opportunity?
approved = mrd[mrd['detailed_status'] == 'Approved'].copy()
approved['req_ht_ldc']  = approved['Requested_Zoning'].apply(ldc_ht)
approved['init_ht_ldc'] = approved['Initial_Zoning'].apply(ldc_ht)
approved['delta_ldc']   = approved.apply(
    lambda r: r['req_ht_ldc'] - r['init_ht_ldc']
    if pd.notna(r['req_ht_ldc']) and pd.notna(r['init_ht_ldc']) else np.nan, axis=1)

appr_valid = approved.dropna(subset=['delta_ldc'])
print(f"=== APPROVED CASES WITH HEIGHT DATA ===")
print(f"  Approved cases total: {len(approved)}")
print(f"  Approved cases with both zones LDC-mapped: {len(appr_valid)}")
if len(appr_valid) > 0:
    a_up   = (appr_valid['delta_ldc'] > 0).sum()
    a_same = (appr_valid['delta_ldc'] == 0).sum()
    a_down = (appr_valid['delta_ldc'] < 0).sum()
    print(f"  req_ht > init_ht (approved upzone across tier): {a_up} ({a_up/len(appr_valid):.1%})")
    print(f"  req_ht == init_ht (same tier approved):         {a_same} ({a_same/len(appr_valid):.1%})")
    print(f"  req_ht < init_ht (approved downzone):           {a_down} ({a_down/len(appr_valid):.1%})")
print()

# The key question: if we had Final_Zoning for all 272 approved cases,
# how many would have a CONCESSION (approved_ht < requested_ht)?
# We can estimate: of the 49 approved cases with both heights,
# what fraction had the approved zone at a lower tier than requested?
appr_fz = approved[approved['Final_Zoning'].fillna('').str.strip() != ''].copy()
appr_fz['approved_ht_ldc'] = appr_fz['Final_Zoning'].apply(ldc_ht)
appr_fz['concession_ldc']  = appr_fz.apply(
    lambda r: max(0, (r['req_ht_ldc'] or 0) - (r['approved_ht_ldc'] or 0))
    if pd.notna(r['req_ht_ldc']) and pd.notna(r['approved_ht_ldc']) else np.nan, axis=1)
print(f"=== LDC HEIGHT CONCESSION (req_ht - approved_ht) for {len(appr_fz)} cases with Final_Zoning ===")
if len(appr_fz) > 0:
    print(appr_fz[['case_number','Initial_Zoning','Requested_Zoning','Final_Zoning',
                   'init_ht_ldc','req_ht_ldc','approved_ht_ldc','concession_ldc']].to_string(index=False))
