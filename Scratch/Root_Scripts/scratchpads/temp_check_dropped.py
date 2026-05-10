import pandas as pd
pets = pd.read_csv(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\recovered_petitions.csv')
pet_cases = set(pets['case_number'].str.strip())

exact_df = pd.read_csv(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\exact_geometric_petition_intensity.csv')
exact_cases = set(exact_df['case_number'].str.strip())

dropped_cases = pet_cases - exact_cases

cm = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv', low_memory=False)
cm['CASE_NUMBER'] = cm['CASE_NUMBER'].str.strip()

dropped_df = cm[cm['CASE_NUMBER'].isin(dropped_cases)].copy()
print(f'Out of {len(dropped_cases)} dropped cases, {len(dropped_df)} are in case_master.')
print(f'Cases with non-null TCAD_ID: {dropped_df["TCAD_ID"].notnull().sum()}')
print(dropped_df[['CASE_NUMBER', 'TCAD_ID']].head(10))
