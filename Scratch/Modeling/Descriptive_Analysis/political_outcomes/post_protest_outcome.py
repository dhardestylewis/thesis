"""
Post-protest outcome analysis:
- For each case in the panel, attach Requested_Zoning and Final_Zoning
  from model_ready to the LAST biweekly period of that case.
- Code direction of zoning change (upgrade / downgrade / overlay-only / unchanged).
- Compare direction between protested and non-protested cases.
"""
import pandas as pd
import numpy as np
import re

PANEL_PATH   = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
MASTER_PATH  = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"

panel  = pd.read_csv(PANEL_PATH,  low_memory=False)
master = pd.read_csv(MASTER_PATH, low_memory=False)

# ── Protest flag from panel (authoritative) ──────────────────────────────────
protested_cases = set(panel[panel['petition_event'] == 1]['case_number'].unique())

# ── Terminal period per case ─────────────────────────────────────────────────
panel['period_start'] = pd.to_datetime(panel['period_start'], errors='coerce')
terminal = (panel
    .sort_values('period_seq')
    .groupby('case_number')
    .last()
    .reset_index()[['case_number','period_start','period_seq']]
    .rename(columns={'period_start': 'terminal_date', 'period_seq': 'terminal_seq'}))

# ── Petition period per case ─────────────────────────────────────────────────
petition_period = (panel[panel['petition_event'] == 1]
    .sort_values('period_seq')
    .groupby('case_number')
    .first()
    .reset_index()[['case_number','period_seq','period_start']]
    .rename(columns={'period_seq': 'petition_seq', 'period_start': 'petition_date'}))

# ── Join final zoning from master ────────────────────────────────────────────
outcomes = master[['case_number','Requested_Zoning','Final_Zoning']].dropna(
    subset=['Requested_Zoning','Final_Zoning'])
outcomes = outcomes.drop_duplicates('case_number')

df = terminal.merge(outcomes, on='case_number', how='inner')
df['protested'] = df['case_number'].isin(protested_cases)
df = df.merge(petition_period, on='case_number', how='left')

# Periods AFTER petition to final resolution (measure of tail length)
df['periods_after_petition'] = df['terminal_seq'] - df['petition_seq']

print(f"Cases with both panel + outcome data: {len(df):,}")
print(f"  Protested:     {df['protested'].sum():,} ({df['protested'].mean()*100:.1f}%)")
print(f"  Non-protested: {(~df['protested']).sum():,}")

# ── Austin zoning intensity hierarchy ────────────────────────────────────────
# Extracts the BASE zoning code (strips overlays like -NP, -CO, -H, -V, -CURE, -NCCD)
INTENSITY = {
    # Rural / Agricultural
    'W': 1, 'RR': 1, 'AG': 1, 'DR': 1,
    # Single Family
    'SF-1': 2, 'SF-2': 2, 'SF-3': 2,
    'SF-4A': 3, 'SF-4B': 3, 'SF-5': 3, 'SF-6': 3,
    # Duplex / Two-Family
    'TF': 3,
    # Multifamily
    'MF-1': 4, 'MF-2': 4, 'MF-3': 5, 'MF-4': 5, 'MF-5': 6, 'MF-6': 6,
    # Office
    'LO': 5, 'GO': 6, 'NO': 5,
    # Neighborhood Commercial
    'LR': 6, 'L': 6,
    # General / Community Commercial
    'GR': 7, 'CS': 7, 'CS-1': 7, 'CR': 7, 'CH': 8,
    # Commercial / Retail / Mixed
    'LI': 8, 'MI': 9, 'HI': 9,
    # High Intensity
    'CBD': 9, 'DMU': 8, 'TOD': 7,
    # Mixed Use (baseline intensity similar to commercial anchor)
    'MU': 7,
    # Special / PUD
    'PUD': 7, 'P': 6,
    # Conditional Overlay as separate base? Usually maps to its base.
    'NO': 5,
}

OVERLAY_STRIP = re.compile(
    r'(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|'
    r'-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-NBG|-SR|-TRN|-PO|-DT|-NO|-OLD)'
)

def base_zone(z):
    if not isinstance(z, str):
        return None
    z = z.strip().upper()
    z = OVERLAY_STRIP.sub('', z)
    return z.strip('-')

def intensity(z):
    b = base_zone(z)
    return INTENSITY.get(b, np.nan)

def overlay_only(req, fin):
    """True if base zones are the same and only overlays changed."""
    return base_zone(req) == base_zone(fin) and req.strip() != fin.strip()

df['req_intensity'] = df['Requested_Zoning'].apply(intensity)
df['fin_intensity'] = df['Final_Zoning'].apply(intensity)
df['zoning_changed'] = df['Requested_Zoning'].str.strip() != df['Final_Zoning'].str.strip()

def direction(row):
    if not row['zoning_changed']:
        return 'Unchanged'
    if overlay_only(str(row['Requested_Zoning']), str(row['Final_Zoning'])):
        return 'Overlay-Only'
    ri, fi = row['req_intensity'], row['fin_intensity']
    if pd.isna(ri) or pd.isna(fi):
        return 'Unknown'
    if fi < ri:
        return 'Downgrade'   # Concession — less intensive than requested
    if fi > ri:
        return 'Upgrade'     # Got MORE than asked (rare but happens)
    return 'Same-Intensity'  # Different code, same tier

df['direction'] = df.apply(direction, axis=1)

print('\n=== Direction of zoning change by protest status ===')
pivot = df.groupby(['protested', 'direction']).size().unstack(fill_value=0)
pivot['Total'] = pivot.sum(axis=1)
for col in pivot.columns[:-1]:
    pivot[f'{col}_pct'] = (pivot[col] / pivot['Total'] * 100).round(1)
print(pivot.to_string())

print('\n=== Downgrade rate (concession) by protest status ===')
print(df.groupby('protested').apply(
    lambda g: pd.Series({
        'downgrade_rate': (g['direction'] == 'Downgrade').mean(),
        'overlay_only_rate': (g['direction'] == 'Overlay-Only').mean(),
        'unchanged_rate': (g['direction'] == 'Unchanged').mean(),
        'n': len(g)
    })
).round(3))

print('\n=== Periods from petition to terminal (panel tail after protest) ===')
prot_tail = df[df['protested'] & df['petition_seq'].notna()]
print(prot_tail['periods_after_petition'].describe().round(1))

print('\n=== Sample protested downgrade cases ===')
sample = df[df['protested'] & (df['direction'] == 'Downgrade')][
    ['case_number','Requested_Zoning','Final_Zoning','req_intensity','fin_intensity','periods_after_petition']
].head(15)
print(sample.to_string())
