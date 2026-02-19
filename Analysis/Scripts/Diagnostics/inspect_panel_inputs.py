"""Quick inspection script to understand data shapes for panel construction planning."""
import json, csv, os, itertools
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data")

# === 1. GeoJSON Property Universe ===
print("=" * 60)
print("1. GeoJSON Property Universe")
print("=" * 60)
geojson_path = os.path.join(DATA_DIR, "Protest_Petitions", "GeoJSON", "protest_petitions_v1.geojson")
with open(geojson_path, 'r') as f:
    data = json.load(f)
feats = data['features']

tcad_std = set()
tcad_raw = set()
case_nums = set()
has_geom = 0
no_geom = 0
for feat in feats:
    p = feat['properties']
    if p.get('standardized_tcad_id'):
        tcad_std.add(p['standardized_tcad_id'])
    if p.get('TCAD ID'):
        tcad_raw.add(p['TCAD ID'])
    if p.get('Case Number'):
        case_nums.add(p['Case Number'])
    if feat.get('geometry'):
        has_geom += 1
    else:
        no_geom += 1

print(f"Total features: {len(feats)}")
print(f"Unique standardized_tcad_id: {len(tcad_std)}")
print(f"Unique TCAD ID (raw): {len(tcad_raw)}")
print(f"Unique Case Numbers: {len(case_nums)}")
print(f"Features with geometry: {has_geom}")
print(f"Features with null geometry: {no_geom}")
print(f"Sample std IDs: {list(itertools.islice(tcad_std, 5))}")
print(f"Sample raw IDs: {list(itertools.islice(tcad_raw, 5))}")

# Year distribution
years_final = []
years_status = []
for feat in feats:
    p = feat['properties']
    fd = p.get('final_date', '')
    sd = p.get('status_date', '')
    if fd and len(fd) >= 4:
        years_final.append(fd[:4])
    if sd and len(sd) >= 4:
        years_status.append(sd[:4])
print(f"\nYear range (final_date): {min(years_final) if years_final else 'N/A'} - {max(years_final) if years_final else 'N/A'}")
print(f"Year range (status_date): {min(years_status) if years_status else 'N/A'} - {max(years_status) if years_status else 'N/A'}")

# === 2. EARS Account Number Format ===
print("\n" + "=" * 60)
print("2. EARS Account Number Samples (2019)")
print("=" * 60)
ears_path = os.path.join(DATA_DIR, "Appraisal_Rolls", "2019", "EARS_2019_Master.csv")
with open(ears_path, 'r', encoding='latin-1') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 5:
            acct = row[6] if len(row) > 6 else "?"
            acct_fmt = row[7] if len(row) > 7 else "?"
            county = row[3] if len(row) > 3 else "?"
            cat = row[30] if len(row) > 30 else "?"
            rec_type = row[0]
            print(f"Row {i}: type={rec_type}, acct={acct!r}, acct_fmt={acct_fmt!r}, county={county!r}, cat={cat!r}")
        else:
            break

# === 3. EARS Files Per Year ===
print("\n" + "=" * 60)
print("3. EARS Files Per Year")
print("=" * 60)
for year in range(2018, 2026):
    d = os.path.join(DATA_DIR, "Appraisal_Rolls", str(year))
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if (f.endswith('.csv') or f.endswith('.txt')) and 'EARS' in f]
        for fn in sorted(files):
            size_mb = os.path.getsize(os.path.join(d, fn)) / (1024*1024)
            print(f"  {year}/{fn}: {size_mb:.0f} MB")

# === 4. How do TCAD IDs in GeoJSON relate to EARS account numbers? ===
print("\n" + "=" * 60)
print("4. TCAD ID formats")
print("=" * 60)
# GeoJSON TCAD IDs
sample_std = list(itertools.islice(tcad_std, 10))
sample_raw = list(itertools.islice(tcad_raw, 10))
print(f"GeoJSON standardized_tcad_id samples: {sample_std}")
print(f"GeoJSON TCAD ID (raw) samples: {sample_raw}")

# EARS account numbers
ears_accts = set()
with open(ears_path, 'r', encoding='latin-1') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 100 and row[0] == 'AJR':
            ears_accts.add(row[6])
        if i > 100:
            break
print(f"EARS account number samples: {list(itertools.islice(ears_accts, 10))}")

# Try matching
overlap = tcad_std.intersection(ears_accts)
print(f"\nDirect overlap (std_tcad vs ears_acct): {len(overlap)}")
# Try stripping dashes from raw TCAD IDs
tcad_nodash = {t.replace('-','') for t in tcad_raw}
overlap2 = tcad_nodash.intersection(ears_accts)
print(f"Overlap (raw TCAD no-dash vs ears_acct): {len(overlap2)}")

# === 5. Land Use Inventory ===
print("\n" + "=" * 60)
print("5. Land Use Inventory Sample")
print("=" * 60)
lui_path = os.path.join(DATA_DIR, "Zoning_Cases", "Source_Data", "land_use_inventory_prefetched.csv")
with open(lui_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Columns: {header}")
    for i, row in enumerate(reader):
        if i < 3:
            print(f"Row {i}: {dict(zip(header, row))}")
        else:
            break

# Count total land use records
with open(lui_path, 'r', encoding='utf-8') as f:
    total = sum(1 for _ in f) - 1
print(f"Total land use records: {total:,}")
