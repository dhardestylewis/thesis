import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths to the AWS outputs (synced locally)
SURFACE_CSV = "output/vae_dose_response_surface_expanded.csv"
OUTPUT_DIR = r"c:\Users\dhl\.gemini\antigravity\brain\1632e32a-ef31-4422-854b-ea7296224fe1\\"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_3d_god_table():
    if not os.path.exists(SURFACE_CSV):
        print(f"Waiting for {SURFACE_CSV} from AWS...")
        return
        
    df = pd.read_csv(SURFACE_CSV)
    
    # Calculate P10, P50, P90 from the flattened MC cases
    grouped = df.groupby('dose')
    sum_df = grouped.agg(
        surv_p50=('surv_mean', 'median'),
        surv_p10=('surv_mean', lambda x: np.percentile(x, 10)),
        surv_p90=('surv_mean', lambda x: np.percentile(x, 90)),
        ht_p50=('ht_mean', 'median'),
        ht_p10=('ht_mean', lambda x: np.percentile(x, 10)),
        ht_p90=('ht_mean', lambda x: np.percentile(x, 90)),
        tok_p50=('tok_mean', 'median')
    ).reset_index()
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Survival Probability", "Height Concession (Raw Ft)", "NLP Tokens (Outrage)", ""))
    
    # SURVIVAL
    fig.add_trace(go.Scatter(x=sum_df['dose'], y=sum_df['surv_p50'], mode='lines+markers', name='Surv Median', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sum_df['dose'], y=sum_df['surv_p90'], fill=None, mode='lines', line_color='rgba(0,0,255,0)', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sum_df['dose'], y=sum_df['surv_p10'], fill='tonexty', mode='lines', line_color='rgba(0,0,255,0)', fillcolor='rgba(0,0,255,0.2)', showlegend=False), row=1, col=1)
    
    # HEIGHT RAW
    fig.add_trace(go.Scatter(x=sum_df['dose'], y=sum_df['ht_p50'], mode='lines+markers', name='Ht Ft Median', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=sum_df['dose'], y=sum_df['ht_p90'], fill=None, mode='lines', line_color='rgba(0,255,0,0)', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=sum_df['dose'], y=sum_df['ht_p10'], fill='tonexty', mode='lines', line_color='rgba(0,255,0,0)', fillcolor='rgba(0,255,0,0.2)', showlegend=False), row=1, col=2)
    
    # TOKENS
    fig.add_trace(go.Scatter(x=sum_df['dose'], y=sum_df['tok_p50'], mode='lines+markers', name='Tokens Median', line=dict(color='orange')), row=2, col=1)

    fig.update_layout(height=800, title_text="Causal Friction Surface (God Table)", template="plotly_dark")
    
    out_path = os.path.join(OUTPUT_DIR, "causal_friction_surface.html")
    fig.write_html(out_path)
    print(f"Generated {out_path}")

def plot_time_series():
    if not os.path.exists(TIME_SERIES_CSV):
        print(f"Waiting for {TIME_SERIES_CSV} from AWS...")
        return
        
    df = pd.read_csv(TIME_SERIES_CSV)
    
    # We want a 3D plot where X=Time, Y=Dose, Z=Metric
    # Plotly 3D Surface
    
    doses = df['dose'].unique()
    times = df['t'].unique()
    
    # Create meshgrid for Surface
    z_ht = np.zeros((len(doses), len(times)))
    
    for i, d in enumerate(doses):
        for j, t in enumerate(times):
            mask = (df['dose'] == d) & (df['t'] == t)
            if mask.sum() > 0:
                z_ht[i, j] = df.loc[mask, 'ht'].values[0]
                
    fig = go.Figure(data=[go.Surface(z=z_ht, x=times, y=doses, colorscale='Viridis')])
    fig.update_layout(title='3D Treatment Effect Over Time (Height Concession)',
                      scene=dict(xaxis_title='Time (Bi-weekly Periods)',
                                 yaxis_title='Petition Dose (0 to 1)',
                                 zaxis_title='Height Lost (ft)'),
                      template="plotly_dark", height=800)
                      
    out_path = os.path.join(OUTPUT_DIR, "causal_time_series_3d.html")
    fig.write_html(out_path)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    plot_3d_god_table()
    plot_time_series()
