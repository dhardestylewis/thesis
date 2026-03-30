import geopandas as gpd
path = r'c:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\GeoJSON\zoning_cases_with_nearby_parcels.geojson'
gdf = gpd.read_file(path, rows=5)
print("COLUMNS:")
print(gdf.columns)
