import pandas as pd, re

df = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Final\model_ready_zoning_data.csv', low_memory=False)

LDC_CODES = {'RR','LA','DR','SF-1','SF-2','SF-3','SF-4A','SF-4B','SF-5','SF-6','MH',
             'MF-1','MF-2','MF-3','MF-4','MF-5','MF-6','NO','LO','GO','CR','LR','GR',
             'CS','CS-1','CH','IP','LI','MI','HI','CBD','DMU','W','P','MU','AG','I'}

ALIAS_MAP = {
    'IRR': 'RR', 'I-RR': 'RR',
    'ISF2': 'SF-2', 'I-SF-2': 'SF-2', 'I-SF2': 'SF-2',
    'ISF3': 'SF-3', 'I-SF-3': 'SF-3',
    'IMF4': 'MF-4', 'I-MF-4': 'MF-4',
    'GRCO': 'GR', 'GR-CO': 'GR', 'GR-MU': 'GR', 'GR-MU-CO': 'GR',
    'CSCO': 'CS', 'CS-CO': 'CS', 'CS-MU': 'CS',
    'LRCO': 'LR', 'LR-CO': 'LR',
}

def extract_base_code(s):
    if pd.isna(s) or str(s).strip() in ('999', '1', ''):
        return None
    s_up = str(s).strip().upper()
    token = s_up.split()[0].rstrip(',')
    if token in ALIAS_MAP:
        return ALIAS_MAP[token]
    if token in LDC_CODES:
        return token
    for code in sorted(LDC_CODES, key=len, reverse=True):
        if s_up == code or s_up.startswith(code + '-') or s_up.startswith(code + ' ') or s_up.startswith(code + ','):
            return code
    for alias, base in ALIAS_MAP.items():
        if s_up.startswith(alias):
            return base
    return None

df['existing_base'] = df['existing_zoning'].apply(extract_base_code)
df['proposed_base'] = df['proposed_zoning'].apply(extract_base_code)

n_ex = df['existing_base'].notna().sum()
n_prop = df['proposed_base'].notna().sum()
n_both = (df['existing_base'].notna() & df['proposed_base'].notna()).sum()
print(f"existing_base extracted: {n_ex} / {len(df)}")
print(f"proposed_base extracted: {n_prop} / {len(df)}")
print(f"Both parseable (can compute delta): {n_both}")

# Fallback: use Initial_Zoning when existing_zoning fails
df['existing_base_v2'] = df['existing_base'].fillna(df['Initial_Zoning'].apply(extract_base_code))
n_both_v2 = (df['existing_base_v2'].notna() & df['proposed_base'].notna()).sum()
print(f"With Initial_Zoning fallback: {n_both_v2}")

# Also check: what fraction of test-set cases (year >= 2019) are covered?
panel = pd.read_csv(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv', low_memory=False)
test_cases = panel[panel['year'] >= 2019]['case_number'].unique()
df_test = df[df['case_number'].isin(test_cases)]
n_test_both = (df_test['existing_base_v2'].notna() & df_test['proposed_base'].notna()).sum()
print(f"\nTest-set cases (year >= 2019): {len(df_test)}")
print(f"Test cases with computable delta: {n_test_both} ({n_test_both/len(df_test)*100:.1f}%)")

# Still failing after alias expansion - top categories
still_bad = df[df['existing_zoning'].notna() & (df['existing_zoning'] != '999') & df['existing_base'].isna()]
print("\nStill unparseable existing_zoning (top 10):")
print(still_bad['existing_zoning'].value_counts().head(10))
