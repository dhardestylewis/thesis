import pandas as pd
import numpy as np
from scipy import stats

OUT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756'
MASTER_PATH = r'C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv'
PET_INTENSITY = rf'{OUT_DIR}\petition_intensity_corrected.csv'

pet = pd.read_csv(PET_INTENSITY)
master = pd.read_csv(MASTER_PATH, low_memory=False)

master['case_number'] = master['case_number'].str.strip()
pet['case_number'] = pet['case_number'].str.strip()

df = master.dropna(subset=['Final_Zoning']).drop_duplicates('case_number').copy()
df['req_co'] = df['Requested_Zoning'].str.contains('-CO', na=False)
df['fin_co'] = df['Final_Zoning'].str.contains('-CO', na=False)
df['z_changed'] = df['Requested_Zoning'].str.strip() != df['Final_Zoning'].str.strip()
df['co_added_penalty'] = (~df['req_co'] & df['fin_co'] & df['z_changed']).astype(float)

df = pet[['case_number', 'true_petition_pct']].merge(df[['case_number', 'co_added_penalty']], on='case_number', how='inner')

def run_rd_cutoff(cutoff, t_col):
    df['run_var'] = df['true_petition_pct'] - cutoff
    rd_df = df[(df['run_var'] >= -15) & (df['run_var'] <= 15)].copy()
    if len(rd_df[rd_df['run_var'] >= 0]) >= 3 and len(rd_df[rd_df['run_var'] < 0]) >= 3:
        left = rd_df[rd_df['run_var'] < 0][t_col].mean()
        right = rd_df[rd_df['run_var'] >= 0][t_col].mean()
        ate = right - left
        pooled_se = np.sqrt(rd_df[rd_df['run_var'] < 0][t_col].var()/max(len(rd_df[rd_df['run_var'] < 0]),1) + rd_df[rd_df['run_var'] >= 0][t_col].var()/max(len(rd_df[rd_df['run_var'] >= 0]),1))
        z_stat = ate / (pooled_se + 1e-9)
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return ate, p_val
    return np.nan, np.nan

print('\nConditional Overlay (-CO) Imposition as a Target:')
print(f"{'Cutoff':<12} | {'Discontinuous Jump (ATE)':<25} | {'P-Value':<10}")
print('-' * 60)
for c in [5, 10, 15, 20, 25, 30]:
    ate, p = run_rd_cutoff(c, 'co_added_penalty')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    mark = ' (LEGAL)' if c == 20 else ''
    print(f"{str(c)+'%'+mark:<12} | {ate:>+10.3f}                       | {p:.3f} ({sig})")
