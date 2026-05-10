import pandas as pd
df_plan = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv')
df_zap = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\zoning_platting_commission_index.csv')
df_all = pd.concat([df_plan, df_zap])
df_all['Case'] = df_all['Doc_Text'].str.extract(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})')
print(f"Cases found in index Doc_Text: {df_all['Case'].notna().sum()} out of {len(df_all)}")
