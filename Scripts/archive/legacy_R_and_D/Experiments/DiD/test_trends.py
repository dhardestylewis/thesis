import pandas as pd
import sys

csv_path = 'C:/Users/dhl/data/thesis/thesis/Data/Zoning_Cases/Processed_Data/CSV/submission_grade_icp_matrix.csv'
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Failed to load {csv_path}: {e}")
    sys.exit()

dist_cols = [c for c in df.columns if 'dist' in c.lower()]
if not dist_cols:
    print("Could not find a district column.")
    print("Columns:", df.columns.tolist()[:20])
    sys.exit()

dist_col = dist_cols[0]
date_col = 'Meeting_Date'
target_col = 'valid_petition'

if target_col not in df.columns:
    target_col = 'protested' # fallback
    if target_col not in df.columns:
        print("Could not find formal petition label column.")
        sys.exit()

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df['Year'] = df[date_col].dt.year
df = df.dropna(subset=['Year', dist_col, target_col])

# Filter strictly to the pre/post window if needed
df = df[(df['Year'] >= 2018) & (df['Year'] <= 2024)]

df['Treated'] = df[dist_col].apply(lambda x: 1 if str(x).split(".")[0] in ['4', '9'] else 0)

yearly = df.groupby(['Year', 'Treated'])[target_col].mean().unstack()
counts = df.groupby(['Year', 'Treated'])[target_col].count().unstack()

print("\n--- MEAN VALID PETITION RATES ---")
print("Year | Control (Cols=0) | Treated D4/D9 (Cols=1)")
print(yearly.round(3))

print("\n--- CASE VOLUME (N) ---")
print("Year | Control (Cols=0) | Treated D4/D9 (Cols=1)")
print(counts)
