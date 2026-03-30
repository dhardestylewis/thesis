import geopandas as gpd
path = r'c:\Users\dhl\data\thesis\thesis\Data\GIS\TCAD\tcad_parcels.geojson'
gdf = gpd.read_file(path, rows=5)
print(gdf.columns)
for col in gdf.columns:
    if 'id' in col.lower() or 'prop' in col.lower():
        print(f"{col}: {gdf[col].iloc[0]}")
