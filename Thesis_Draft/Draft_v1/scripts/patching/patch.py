import re
import os

fp_did = r'c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Modeling\Production_Models\run_causal_track3_did_real.py'
fp_f17 = r'C:\Users\dhl\data\Thesis\thesis\scratch\plots\scratch_plot_did_event_study.py'

for fp in [fp_did, fp_f17]:
    if not os.path.exists(fp):
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        t = f.read()
    
    t = re.sub(r'df_votes\s*=\s*pd\.read_csv\(VOTE_DATA.*?\n', '', t)
    t = re.sub(r'df_votes\s*=\s*df_votes\.groupby.*?\n', '', t)
    t = re.sub(r'df\s*=\s*df_h0\.merge\(df_votes.*?\n', 'df = df_h0.copy()\n', t)
    
    t = t.replace('zoning_code', 'ldb_basezone')
    t = t.replace("'SF-'", "'SF|MF|PUD|TND'")
    
    t = t.replace('vote_no', 'is_protested')
    t = t.replace('Vote_No Outcome', 'Organized Opposition')
    t = t.replace('Council Vote_No Magnitude', 'Organized Opposition')
    t = t.replace('Council Dissent', 'Organized Opposition')
    
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(t)
