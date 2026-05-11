import pandas as pd, re, os

IDX_DIR = r'c:\Users\dhl\data\Thesis\thesis\Data\raw\indices'
plan = pd.read_csv(os.path.join(IDX_DIR, 'planning_commission_index.csv'), low_memory=False)
zap  = pd.read_csv(os.path.join(IDX_DIR, 'zoning_platting_commission_index.csv'), low_memory=False)
idx  = pd.concat([plan, zap], ignore_index=True)

print('Index year range:', idx['Year'].min(), '-', idx['Year'].max())
print()
print('Index doc counts per year:')
print(idx['Year'].value_counts().sort_index().to_string())
print()

# What do URLs look like across years?
print('Sample Doc_URLs by year:')
for yr in [2009, 2010, 2015, 2020, 2025]:
    rows = idx[idx['Year'] == yr]
    if len(rows) > 0:
        url = rows['Doc_URL'].iloc[0]
        print(f'  {yr}: {url}')
print()

# Are years before the index start simply not gathered?
idx_years = set(idx['Year'].unique())
print(f'Index covers years: {sorted(idx_years)[:5]} ... {sorted(idx_years)[-3:]}')
print()

# Check what the commission transcripts corpus actually covers year-wise
comm = pd.read_csv(
    r'c:\Users\dhl\data\Thesis\thesis\Data\interim\commission_transcripts.csv',
    low_memory=False, usecols=['Filename'])
def fn_year(fn):
    m = re.match(r'^(\d{4})_', str(fn))
    return int(m.group(1)) if m else None
comm['year'] = comm['Filename'].apply(fn_year)
print('Commission transcripts year range:', comm['year'].min(), '-', comm['year'].max())
print('Counts per year:')
print(comm['year'].value_counts().sort_index().to_string())
print()

# So: missing-zone cases created pre-2009 = simply not in our scrape at all
# Confirm by checking the missing cases' year distribution vs index start
mrd = pd.read_csv(
    r'c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv',
    low_memory=False)
missing = mrd[mrd['Requested_Zoning'].fillna('').str.strip() == '']
before_idx = missing[missing['calendar_year_folder_created'] < idx['Year'].min()].shape[0]
after_idx  = missing[missing['calendar_year_folder_created'] >= idx['Year'].min()].shape[0]
print(f'Missing-zone cases created BEFORE index start ({idx["Year"].min()}): {before_idx}')
print(f'Missing-zone cases created AFTER index start:                        {after_idx}')
print()
print('=> Cases before index start simply were never scraped.')
print('   Need to find the pre-2009 URL scheme or archive source.')
print()

# Check if the City has a Legistar or earlier archive
# Look at what the index URLs look like — are they all EDIMS?
url_patterns = idx['Doc_URL'].dropna().apply(lambda u: re.match(r'https?://([^/]+)', str(u)).group(1) if re.match(r'https?://([^/]+)', str(u)) else 'unknown')
print('URL host breakdown:')
print(url_patterns.value_counts().to_string())
