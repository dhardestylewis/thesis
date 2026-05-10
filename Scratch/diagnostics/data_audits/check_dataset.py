import pandas as pd
df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv', usecols=['Core_Case', 'parcel_id_10'])
print(f'Total Rows: {len(df)}')
print(f'Unique Cases: {df["Core_Case"].nunique()}')
npa_rows = df[df["Core_Case"] == "NPA-2017-0005"]
print(f'NPA-2017-0005 rows: {len(npa_rows)}')
