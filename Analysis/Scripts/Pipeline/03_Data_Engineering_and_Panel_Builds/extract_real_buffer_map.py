import json
import csv
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
GEOJSON_PATH = os.path.join(ROOT, "Data", "Protest_Petitions", "GeoJSON", "protest_petitions_v1.geojson")
OUT_PATH = os.path.join(ROOT, "Data", "Warehouse_As_Of", "Build", "case_buffer_map.csv")

def extract():
    print("Loading GeoJSON...")
    with open(GEOJSON_PATH, 'r') as f:
        data = json.load(f)
    
    mapping = []
    print(f"Processing {len(data['features'])} features...")
    
    for feat in data['features']:
        p = feat['properties']
        case = p.get('Case Number')
        tcad = p.get('standardized_tcad_id')
        
        if case and tcad:
            mapping.append({'CASE_NUMBER': case.strip().upper(), 'neighbor_tcad_id': tcad.strip()})
            
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['CASE_NUMBER', 'neighbor_tcad_id'])
        writer.writeheader()
        writer.writerows(mapping)
        
    print(f"Successfully extracted {len(mapping)} case-neighbor mappings to {OUT_PATH}")

if __name__ == "__main__":
    extract()
