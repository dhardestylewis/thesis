import pandas as pd

try:
    h = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical\H0_Filing.csv')
    cross = pd.read_csv(r'C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\enriched_zoning_data_causal.csv', usecols=['Case Number', 'TCAD ID'])
    cross['case_number']=cross['Case Number'].astype(str).str.strip().str.upper()
    cross=cross.drop_duplicates('case_number')
    cross['TCAD ID']=cross['TCAD ID'].astype(str).str.replace(r'[- ]', '', regex=True).str.lstrip('0')
    
    df=h.merge(cross[['case_number', 'TCAD ID']], on='case_number', how='left')
    print('1. Valid TCAD IDs joining from ZONING_CAUSAL into H0:', (df['TCAD ID']!='nan').sum(), 'out of', len(df))
    
    df['year'] = pd.to_numeric(df['case_number'].str.extract(r'C\d+[A-Z]*-(\d{4})')[0], errors='coerce')
    df['year'] = df['year'].fillna(pd.to_numeric(df['case_number'].str.extract(r'((?:19|20)\d\d)')[0], errors='coerce'))
    df['year'] = df['year'].fillna(2020)
    
    p = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Panel\parcel\property_universe.csv', usecols=['standardized_tcad_id', 'year'], low_memory=False)
    p['standardized_tcad_id'] = p['standardized_tcad_id'].astype(str).str.replace(r'[- ]', '', regex=True).str.lstrip('0')
    
    print('   Panel size:', len(p), 'Unique years in panel:', p['year'].unique())
    
    df_keys = set(zip(df[df['TCAD ID']!='nan']['TCAD ID'], df[df['TCAD ID']!='nan']['year']))
    p_keys = set(zip(p['standardized_tcad_id'], p['year']))
    
    matched = df_keys.intersection(p_keys)
    print('2. Successful Panel Matches (TCAD ID + Year):', len(matched))
    
    # What if we just match on TCAD ID without year?
    df_tcad_only = set(df[df['TCAD ID']!='nan']['TCAD ID'])
    p_tcad_only = set(p['standardized_tcad_id'])
    
    print('3. Successful Matches ignoring Year (TCAD ID only):', len(df_tcad_only.intersection(p_tcad_only)))
    
except Exception as e:
    print("Error:", e)
