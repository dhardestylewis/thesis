import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

def get_empirical_cases(freq):
    print("Loading Empirical Cases...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    master = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", low_memory=False)
    import re
    OVERLAY_STRIP = re.compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
    INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
    def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)
    master["case_number"] = master["case_number"].astype(str).str.strip()
    master["req_int"] = master["Requested_Zoning"].apply(get_int)
    master["fin_int"] = master["Final_Zoning"].apply(get_int)
    master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
    master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)
    
    df["case_number"] = df["case_number"].astype(str).str.strip()
    df = df.merge(master[["case_number", "t_downgrade"]].drop_duplicates("case_number"), on="case_number", how="left")
    df["t_downgrade"] = df["t_downgrade"].fillna(0)
    
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    df["council_hearings_this_period"] = df["council_hearings_this_period"].fillna(0).astype(float)
    df["commission_hearings_this_period"] = df["commission_hearings_this_period"].fillna(0).astype(float)
    
    # We only care about cases that actually HAD a protest
    treated = df[df['cumulative_petition_pct'] > 0]
    
    records = []
    groups = treated.groupby('case_number')
    for case, group in groups:
        group = group.sort_values('period_seq')
        # find intervention timing
        intervention_idx = group[group['petition_pct_this_period'] > 0].index
        if len(intervention_idx) == 0: continue
        
        first_protest = group.loc[intervention_idx[0]]
        timing = first_protest['period_seq']
        
        max_pct = group['cumulative_petition_pct'].max()
        case_date = pd.to_datetime(group['period_start'].iloc[-1])
        
        records.append({
            'case_date': case_date,
            'timing': timing,
            'intensity': max_pct,
            'survival': group['resolved'].max(), # 1 if died, wait resolved means passed. 0 if died.
            'downgrade': group['t_downgrade'].max(),
            'commission': group['commission_hearings_this_period'].sum(),
            'council': group['council_hearings_this_period'].sum()
        })
        
    emp = pd.DataFrame(records)
    # Flip survival so 1 = survival, 0 = death (Wait, resolved=1 means passed (survived). resolved=0 means died/withdrawn.)
    # In my LSTM logic, Survival = prod(1 - hazard).
    
    # Floor dates to frequency
    emp['era_cutoff'] = emp['case_date'].dt.to_period(freq).dt.end_time
    return emp

def replot_target(target, freq, emp_df, title, z_title, out_filename):
    print(f"Re-plotting {target} at {freq} frequency...")
    master_csv = os.path.join(OUT_DIR, f'master_surfaces_{target}.csv')
    df = pd.read_csv(master_csv)
    
    df['era_cutoff'] = pd.to_datetime(df['era_cutoff'])
    
    # Downsample
    # Group by (frequency, period_seq, petition_pct) -> mean z_outcome
    df['era_freq'] = df['era_cutoff'].dt.to_period(freq).dt.end_time
    
    downsampled = df.groupby(['era_freq', 'period_seq', 'petition_pct'])['z_outcome'].mean().reset_index()
    
    eras = sorted(downsampled['era_freq'].unique())
    eras_str = [e.strftime('%Y-%m-%d') for e in eras]
    
    pcts = sorted(downsampled['petition_pct'].unique())
    periods = sorted(downsampled['period_seq'].unique())
    
    global_z_max = downsampled['z_outcome'].max()
    if target in ['survival', 'downgrade']: global_z_max = 1.0
    
    fig = go.Figure()
    colorscale = 'Magma' if target != "downgrade" else 'Viridis'
    
    # For slider logic: we have 2 traces per era (Surface, and Scatter3D)
    # Total traces = 2 * len(eras)
    
    for idx, era in enumerate(eras):
        era_data = downsampled[downsampled['era_freq'] == era]
        Z = era_data.pivot(index='petition_pct', columns='period_seq', values='z_outcome').values
        
        is_visible = (idx == len(eras) - 1)
        
        # 1. Add Surface
        fig.add_trace(go.Surface(
            z=Z, x=periods, y=pcts,
            colorscale=colorscale,
            cmin=0, cmax=global_z_max,
            colorbar=dict(title=z_title),
            visible=is_visible,
            name=f"Surface ≤ {eras_str[idx]}",
            showscale=True,
            opacity=0.9
        ))
        
        # 2. Add Empirical Scatter
        # All cases that happened UP TO this era
        emp_era = emp_df[emp_df['era_cutoff'] <= era]
        
        fig.add_trace(go.Scatter3d(
            x=emp_era['timing'],
            y=emp_era['intensity'],
            z=emp_era[target],
            mode='markers',
            marker=dict(
                size=4,
                color='cyan' if target in ['survival', 'downgrade'] else 'red',
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            visible=is_visible,
            name=f"Empirical Support (n={len(emp_era)})",
            hovertemplate="Timing: %{x}<br>Intensity: %{y}%<br>Actual Outcome: %{z}"
        ))

    steps = []
    for i, era in enumerate(eras):
        # We need to set exactly 2 traces visible for this step
        visible_array = [False] * (2 * len(eras))
        visible_array[i*2] = True
        visible_array[i*2 + 1] = True
        
        step = dict(
            method="update",
            args=[
                {"visible": visible_array},
                {"title": f'{title}<br><sup>Multi-Task Network | Smoothed: <b>{freq}</b> | Era <b>≤ {eras_str[i]}</b></sup>'}
            ],
            label=f"≤ {eras_str[i]}"
        )
        steps.append(step)

    sliders = [dict(
        active=len(eras) - 1,
        currentvalue={"prefix": "Training Era Cutoff: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders,
        title=f'{title}<br><sup>Multi-Task Network | Smoothed: <b>{freq}</b> | Era <b>≤ {eras_str[-1]}</b></sup>',
        scene=dict(
            xaxis=dict(title='Intervention Timing (Period)', range=[1, 15]),
            yaxis=dict(title='Petition Intensity (%)', range=[0, 100]),
            zaxis=dict(title=z_title, range=[0, global_z_max]),
        ),
        width=1200,
        height=900,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    out_path = rf"{OUT_DIR}\{out_filename}"
    fig.write_html(out_path)
    print(f"  > Saved Smoothed/Overlaid artifact to {out_path}")


def main():
    freq = 'M' # Monthly Smoothing
    emp_df = get_empirical_cases(freq)
    
    replot_target("survival", freq, emp_df, 'The "Gravity Well" of the Supermajority Law', 'Terminal Survival Probability', 'causal_lstm_monthly_overlay_survival.html')
    replot_target("downgrade", freq, emp_df, 'The Zoning Downgrade Hazard', 'Probability of Downzoning', 'causal_lstm_monthly_overlay_downgrade.html')
    replot_target("commission", freq, emp_df, 'The Early Friction Surface', 'Cumulative Hearings', 'causal_lstm_monthly_overlay_commission.html')
    replot_target("council", freq, emp_df, 'The Late Political Friction Surface', 'Cumulative Hearings', 'causal_lstm_monthly_overlay_council.html')

if __name__ == '__main__':
    main()
