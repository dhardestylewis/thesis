import pandas as pd
import re

# Load master zoning cases
df_master = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv')

# Extract year from application_start_date
df_master['Year'] = pd.to_datetime(df_master['application_start_date'], errors='coerce').dt.year
df_master = df_master.dropna(subset=['Year', 'case_number'])
df_master['Year'] = df_master['Year'].astype(int)

# Normalize master case numbers (e.g., C14-2015-0123)
# To handle edge cases where the agenda might omit the extension (.SH, .CO), we can match the core C14 string
def get_core_case(case_str):
    case_str = str(case_str).upper().strip()
    match = re.search(r'(C14(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', case_str)
    if match:
        return match.group(1)
    return case_str

df_master['Core_Case'] = df_master['case_number'].apply(get_core_case)

# Load scraped agenda cases
df_agenda = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/council_agendas_cases.csv')
df_agenda['Core_Case'] = df_agenda['Case_Number'].apply(get_core_case)

# Get the set of unique core cases present in the agendas
agenda_cases_set = set(df_agenda['Core_Case'].unique())

# Compute overlap year by year (from 2009 to 2026)
results = []
for year in range(2009, 2027):
    # Get all master cases for this year
    master_year_cases = set(df_master[df_master['Year'] == year]['Core_Case'].unique())
    total_master = len(master_year_cases)
    
    if total_master == 0:
        continue
        
    # Count how many are in the agenda
    overlap_cases = master_year_cases.intersection(agenda_cases_set)
    overlap_count = len(overlap_cases)
    
    pct = (overlap_count / total_master) * 100
    
    results.append({
        'Year': year,
        'Master_Cases': total_master,
        'Found_in_Agenda': overlap_count,
        'Coverage_%': f"{pct:.1f}%"
    })

# Total overlap
modern_master = df_master[df_master['Year'] >= 2009]
total_modern_cases = modern_master['Core_Case'].nunique()
total_overlap = len(set(modern_master['Core_Case'].unique()).intersection(agenda_cases_set))
total_pct = (total_overlap / total_modern_cases) * 100 if total_modern_cases > 0 else 0

results.append({
    'Year': 'TOTAL (2009+)',
    'Master_Cases': total_modern_cases,
    'Found_in_Agenda': total_overlap,
    'Coverage_%': f"{total_pct:.1f}%"
})

df_results = pd.DataFrame(results)

print("--- OVERLAP ANALYSIS: MASTER ZONING CASES vs CITY COUNCIL AGENDAS ---")
print(df_results.to_markdown(index=False))
