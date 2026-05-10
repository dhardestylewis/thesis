import pandas as pd, os

out_path = 'Data/Panel/geo/case_geoid_lookup.csv'
if os.path.exists(out_path):
    gl = pd.read_csv(out_path)
    matched = gl['geoid_tract'].notna().sum()
    print(f'Geocoder saved to disk: {len(gl)} cases tested, {matched} matched.')
else:
    print('No file saved.')
