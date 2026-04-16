import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_H0 = r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical\H0_Filing_Master_Enriched_v2.csv'
df = pd.read_csv(DATA_H0, low_memory=False)
df['is_protested'] = df['is_protested'].fillna(0)
if 'zoning_code' in df.columns:
    df['is_residential'] = df['zoning_code'].astype(str).str.contains('SF|MF|PUD|TND', na=False).astype(int)

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['quarter'] = (df.index % 4) + 1
df['time_t'] = df['year'] + (df['quarter'] - 1) / 4.0
home1_time = 2024.0
df['rel_time_home1'] = np.floor((df['time_t'] - home1_time) * 4)

leads_lags = []
for k in range(-4, 5):
    if k != -1: 
        prefix = 'm' if k < 0 else 'p'
        num = abs(k)
        col = 'leadlag_h1_' + prefix + str(num)
        df[col] = (df['rel_time_home1'] == k).astype(int) * df['is_residential']
        leads_lags.append(col)

formula_vars = ' + '.join(leads_lags) + ' + is_residential + C(time_t)'
mod_dyn = smf.ols('is_protested ~ ' + formula_vars, data=df)
res_dyn = mod_dyn.fit(cov_type='HC1')

table_lines = ['| Quarter ($t$) | Coefficient | Std. Error | P-Value |', '|----|----|----|----|']
for k in range(-4, 5):
    if k == -1:
        table_lines.append('| $t - 1$ (Baseline) | exactly `0.000` | `0.000` | - |')
    else:
        prefix = 'm' if k < 0 else 'p'
        col = 'leadlag_h1_' + prefix + str(abs(k))
        val_k = '$t ' + str(k) + '$' if k < 0 else '$t + ' + str(k) + '$' if k > 0 else '$t=0$'
        coef = res_dyn.params[col]
        se = res_dyn.bse[col]
        pval = res_dyn.pvalues[col]
        table_lines.append(f'| {val_k} | {coef:.3f} | {se:.3f} | {pval:.3f} |')

with open('did_final_table.md', 'w') as f:
    f.write('\n'.join(table_lines))
