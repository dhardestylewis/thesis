import os, shutil

def mv(src, dst):
    if not os.path.exists(src):
        print('SKIP: ' + os.path.basename(src))
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(os.path.basename(src) + ' -> ' + os.path.relpath(dst, os.path.dirname(os.path.dirname(src))))

# ── Warehouse_As_Of ───────────────────────────────────────────────────────────
WAR = r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of'
os.makedirs(os.path.join(WAR, 'canonical'),     exist_ok=True)  # production files used for modeling
os.makedirs(os.path.join(WAR, 'superceded'),    exist_ok=True)  # older builds replaced by v2
os.makedirs(os.path.join(WAR, 'pipeline'),      exist_ok=True)  # H1/H2/H3 intermediate stages
os.makedirs(os.path.join(WAR, 'audit'),         exist_ok=True)  # provenance/attribution

# Canonical production datasets (v2 = latest)
for f in ['H0_Filing_Master_Enriched_v2.csv',
          'H0_Filing_Master_Enriched_v2_OmniLagged.csv',
          'H0_Filing.csv',
          'H0_Filing_Complete.csv']:
    mv(os.path.join(WAR, f), os.path.join(WAR, 'canonical', f))

# Superceded v1 builds
for f in ['H0_Filing_Master_Enriched.csv',
          'H0_Filing_Master_Enriched_Lagged.csv',
          'H0_Filing_Master_Enriched_OmniLagged.csv']:
    mv(os.path.join(WAR, f), os.path.join(WAR, 'superceded', f))

# Intermediate pipeline stages
for f in ['H1_Notice.csv', 'H2_Pre_Commission.csv',
          'H3_Filing_Master_NLP.csv', 'H3_Pre_Council.csv']:
    mv(os.path.join(WAR, f), os.path.join(WAR, 'pipeline', f))

# Audit / provenance
mv(os.path.join(WAR, 'Feature_Provenance_Audit.json'),  os.path.join(WAR, 'audit', 'Feature_Provenance_Audit.json'))
mv(os.path.join(WAR, 'Meta_Attribution_PostClustered.csv'), os.path.join(WAR, 'audit', 'Meta_Attribution_PostClustered.csv'))

print('Warehouse_As_Of:', sorted(os.listdir(WAR)))

# ── Panel ─────────────────────────────────────────────────────────────────────
PAN = r'C:\Users\dhl\data\Thesis\thesis\Data\Panel'
os.makedirs(os.path.join(PAN, 'geo'),       exist_ok=True)
os.makedirs(os.path.join(PAN, 'parcel'),    exist_ok=True)
os.makedirs(os.path.join(PAN, 'census'),    exist_ok=True)

# Geo lookup tables
mv(os.path.join(PAN, 'case_geoid_lookup.csv'),  os.path.join(PAN, 'geo', 'case_geoid_lookup.csv'))

# Parcel-level aggregations
for f in ['case_parcel_agg.csv', 'case_parcel_agg_v2.csv', 'property_universe.csv']:
    mv(os.path.join(PAN, f), os.path.join(PAN, 'parcel', f))

# Census time-series
for f in ['acs_tract_timeseries.csv', 'census_tract_timeseries.csv']:
    mv(os.path.join(PAN, f), os.path.join(PAN, 'census', f))

print('Panel:', sorted(os.listdir(PAN)))
