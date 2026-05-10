"""
Estimate Magnitude of Zoning Concessions
Calculates the human-readable impact of zoning downgrades by mapping
the requested vs final zoning codes to their Austin Land Development Code default heights.
"""
import pandas as pd
import numpy as np
import re

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity_corrected.csv"

master = pd.read_csv(MASTER_PATH, low_memory=False)
pet = pd.read_csv(PET_INTENSITY)

# Standardize IDs and merge
master['case_number'] = master['case_number'].str.strip()
pet['case_number'] = pet['case_number'].str.strip()

# We only care about cases that reached Final Zoning
df = master.dropna(subset=['Final_Zoning']).drop_duplicates('case_number').copy()
df = df.merge(pet[['case_number', 'label_valid_protest']], on='case_number', how='left')
df['label_valid_protest'] = df['label_valid_protest'].fillna(0)

# 1. Base District Heights (Austin LDC Chapter 25-2 Subchapter C)
# Base maximum heights in feet for primary zoning codes
BASE_HEIGHTS = {
    "SF-1": 35, "SF-2": 35, "SF-3": 35, "SF-4A": 35, "SF-4B": 35, "SF-5": 35, "SF-6": 35,
    "TF": 35, "RR": 35, "LA": 35,
    "MF-1": 40, "MF-2": 40, "MF-3": 40, 
    "MF-4": 60, "MF-5": 60, "MF-6": 90,
    "NO": 35, "LO": 40, "GO": 60, 
    "LR": 40, "GR": 60, "CS": 60, "CS-1": 60, "CG": 60, "CR": 60, "CH": 60,
    "LI": 60, "MI": 90, "HI": 90,
    "CBD": 120, "DMU": 120, "TOD": 60, "MU": 60, "PUD": 60
}

OVERLAY_STRIP = re.compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")

def get_base_height(z):
    if not isinstance(z, str): return np.nan
    base = OVERLAY_STRIP.sub("", z.strip().upper()).strip("-")
    return BASE_HEIGHTS.get(base, np.nan)

df['req_base_height'] = df['Requested_Zoning'].apply(get_base_height)
df['fin_base_height'] = df['Final_Zoning'].apply(get_base_height)

# Calculate implied height concession
df['height_concession'] = df['req_base_height'] - df['fin_base_height']
df['z_changed'] = df['Requested_Zoning'].str.strip() != df['Final_Zoning'].str.strip()

# 2. Conditional Overlays (-CO)
# Often used to cap height below the base district limit.
df['req_co'] = df['Requested_Zoning'].str.contains('-CO', na=False)
df['fin_co'] = df['Final_Zoning'].str.contains('-CO', na=False)
df['co_added_penalty'] = (~df['req_co'] & df['fin_co'] & df['z_changed']).astype(int)

# Identify cases that took ANY structural downgrade (base height drop OR added a restrictive CO)
df['structural_downgrade'] = ((df['height_concession'] > 0) | (df['co_added_penalty'] == 1)).astype(int)

downgrades = df[df['structural_downgrade'] == 1].copy()

print("="*60)
print("MAGNITUDE OF CONCESSIONS: WHAT DOES A DOWNGRADE MEAN IN REALITY?")
print("="*60)

print("\n1. How are Developers Forced to Compromise?")
mech = downgrades.groupby('label_valid_protest').agg(
    n_concessions=('case_number', 'count'),
    pct_taking_base_height_drop=('height_concession', lambda x: (x > 0).mean()),
    avg_base_height_lost_ft=('height_concession', lambda x: x[x>0].mean()), # only among those that lost base height
    pct_forced_into_conditional_overlay=('co_added_penalty', 'mean')
).round(2)
mech.index = ["Normal Downgrades (No Valid Protest)", "Protest-Induced Downgrades (>=20%)"]
print(mech.to_string())

print("\n2. Examples of Protest-Induced Height Concessions:")
protest_height_drops = downgrades[(downgrades['label_valid_protest']==1) & (downgrades['height_concession'] > 0)]
cols_to_show = ['case_number', 'Requested_Zoning', 'Final_Zoning', 'req_base_height', 'fin_base_height', 'height_concession']
print(protest_height_drops[cols_to_show].head(10).to_string())
