import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os

# Load the data
df = pd.read_csv("vae_dose_response_surface_expanded.csv")

# Function to create and save a dose-response plot
def create_and_open_plot(metric, y_label, title, filename, color):
    fig = go.Figure()

    # Add the uncertainty band (p10 to p90)
    fig.add_trace(go.Scatter(
        x=df['dose'].tolist() + df['dose'].tolist()[::-1],
        y=df[f'{metric}_p90'].tolist() + df[f'{metric}_p10'].tolist()[::-1],
        fill='toself',
        fillcolor=f'rgba({color},0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Epistemic Uncertainty (10th-90th Percentile)'
    ))

    # Add the median line (p50)
    fig.add_trace(go.Scatter(
        x=df['dose'],
        y=df[f'{metric}_p50'],
        line=dict(color=f'rgb({color})', width=3),
        mode='lines+markers',
        name='Median (50th Percentile)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Intervention Dose (Petition %)",
        yaxis_title=y_label,
        template="plotly_dark",
        hovermode="x unified",
        font=dict(family="Inter, sans-serif")
    )
    
    filepath = os.path.abspath(filename)
    fig.write_html(filepath)
    webbrowser.open_new_tab(f"file://{filepath}")

# Create the 3 plots
create_and_open_plot('surv', 'Resolution Probability', 'Causal Dose-Response: Case Survival', 'surv_response.html', '0, 150, 255')
create_and_open_plot('ht', 'Proposed Max Height (ft)', 'Causal Dose-Response: Building Height Friction', 'ht_response.html', '255, 100, 100')
create_and_open_plot('tok', 'Council Transcript Length (Tokens)', 'Causal Dose-Response: Council Hearing Friction', 'tok_response.html', '150, 255, 150')

print("Generated and opened 3 HTML plots.")
