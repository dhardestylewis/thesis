import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv")

def calculate_units(row, phase):
    zoning_col = f'{phase}_Zoning'
    zoning = str(row.get(zoning_col, '')).upper()
    sqft_col = f'{phase}_Max_SqFt'
    lot_col = f'{phase}_min_lot_sqft'
    area = row.get('shape_area', 0)
    
    if pd.isna(area) or area == 0:
        return np.nan
        
    # If it's a Single Family base zone (SF, RR, LA, DR)
    if any(zoning.startswith(prefix) for prefix in ['SF', 'RR', 'LA', 'DR']):
        min_lot = row.get(lot_col)
        if pd.isna(min_lot) or min_lot == 0:
            return np.nan
        # Units restricted by minimum lot size
        return area / min_lot
    else:
        # For Multi-Family, Commercial, CBD, etc., FAR dictates volume.
        # Assuming an average gross apartment unit size of 1,000 sqft (including common areas).
        max_sqft = row.get(sqft_col)
        if pd.isna(max_sqft) or max_sqft == 0:
            return np.nan
        return max_sqft / 1000.0

print("Calculating Developer Yield Metrics...")
df['Developer_Requested_Units'] = df.apply(lambda row: calculate_units(row, 'Requested'), axis=1)
df['Developer_Approved_Units'] = df.apply(lambda row: calculate_units(row, 'Approved'), axis=1)

# The total number of homes/apartments erased by NIMBYs
df['Unit_Yield_Attrition'] = df['Developer_Requested_Units'] - df['Developer_Approved_Units']

print("Sample of Unit Yield Attrition:")
print(df[['case_number', 'Requested_Zoning', 'Final_Zoning', 'Developer_Requested_Units', 'Developer_Approved_Units', 'Unit_Yield_Attrition']].dropna(subset=['Unit_Yield_Attrition']).head(15))

df.to_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv", index=False)
print("Saved Developer Metrics to CSV.")
