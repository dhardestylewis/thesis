import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

def generate_attrition():
    print("[*] Generating Pre-Council Attrition Timeline (Table 12)...")
    df_h0 = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
    df_h3 = pd.read_csv(os.path.join(DATA, 'H3_Pre_Council.csv'), low_memory=False)

    df_h0['year'] = pd.to_numeric(df_h0['year'], errors='coerce')
    df_h0 = df_h0.dropna(subset=['year']).copy()
    
    target = 'is_protested' if 'is_protested' in df_h0.columns else 'protest'
    df_h0['is_protested'] = pd.to_numeric(df_h0[target], errors='coerce').fillna(0).astype(int)

    df_h3['year'] = pd.to_numeric(df_h3['year'], errors='coerce')
    
    df_h0['case_number'] = df_h0['case_number'].astype(str).str.strip().str.upper()
    df_h3['case_number'] = df_h3['case_number'].astype(str).str.strip().str.upper()
    
    h3_cases_set = set(df_h3['case_number'].dropna())

    # Compute attrition by determining if case_number is in H3_Pre_Council
    df_h0['attrited'] = ~df_h0['case_number'].isin(h3_cases_set)

    timeline = df_h0.groupby('year').agg(
        Total_Filed=('case_number', 'count'),
        Attrited=('attrited', 'sum'),
        Opposed_Filed=('is_protested', 'sum'),
    ).reset_index()

    timeline['Attrition_Rate'] = timeline['Attrited'] / timeline['Total_Filed']
    opposed_attrited = df_h0[df_h0['is_protested'] == 1].groupby('year')['attrited'].sum().rename('Opposed_Attrited')
    timeline = timeline.merge(opposed_attrited, on='year', how='left').fillna({'Opposed_Attrited': 0})

    # Build TeX Table
    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{\textbf{Pre-Council Attrition Timeline:} Opposed vs.\ Unopposed Zoning Cases. (Longitudinal break-down showing structural stability of attrition ratios across shifting policy periods).}')
    tex.append(r'\label{tab:chilling_effect}')
    tex.append(r'\resizebox{0.85\columnwidth}{!}{%')
    tex.append(r'\begin{tabular}{lrrrrr}')
    tex.append(r'\toprule')
    tex.append(r'\textbf{Year} & \textbf{Total Filed} & \textbf{Total Attrited} & \textbf{Total Attr\_Rate (\%)} & \textbf{Opposed Filed} & \textbf{Opposed Attr\_Rate (\%)} \\')
    tex.append(r'\midrule')

    for _, row in timeline.iterrows():
        if row['year'] < 2012 or row['year'] > 2024: continue
        yr = int(row['year'])
        t_f = int(row['Total_Filed'])
        t_a = int(row['Attrited'])
        t_ar = f"{row['Attrition_Rate']*100:.1f}\\%"
        o_f = int(row['Opposed_Filed'])
        o_ar = f"{(row['Opposed_Attrited']/o_f)*100:.1f}\\%" if o_f > 0 else "0.0\\%"
        
        tex.append(f"{yr} & {t_f} & {t_a} & {t_ar} & {o_f} & {o_ar} \\\\")

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'}')
    tex.append(r'\end{table}')

    out_dir = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Tables')
    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, 'Table12_Attrition_Timeline.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex))

    print(f'    [+] Saved Table 12 to {tex_path}')

if __name__ == '__main__':
    generate_attrition()
