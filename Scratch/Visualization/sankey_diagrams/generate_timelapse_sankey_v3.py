import plotly.graph_objects as go
import pandas as pd
import numpy as np

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

df['Year'] = pd.to_datetime(df['application_start_date'], errors='coerce').dt.year
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)
df_modern = df[(df['Year'] >= 2009) & (df['Year'] <= 2024)]

years = sorted(df_modern['Year'].unique())

labels = [
    "Application Filed",              # 0
    "Planning Commission (PC)",       # 1
    "Zoning & Platting (ZAP)",        # 2
    "Unresolved / Failed to Pass",    # 3
    "City Council Agenda",            # 4
    "Approved / Ordinance Passed",    # 5
    " ",                              # 6 (Dummy Source)
    " "                               # 7 (Dummy Target)
]

node_x = [0.05, 0.35, 0.35, 0.95, 0.65, 0.95, 0.0, 1.0]
node_y = [0.05, 0.05, 0.60, 0.85, 0.05, 0.05, 1.0, 1.0]

colors = [
    "#34495e", # App
    "#2980b9", # PC
    "#3498db", # ZAP
    "#e74c3c", # Unresolved
    "#8e44ad", # Council
    "#2ecc71", # Approved
    "rgba(0,0,0,0)", # Dummy
    "rgba(0,0,0,0)"  # Dummy
]

max_volume = 0
for year in years:
    v = len(df_modern[df_modern['Year'] == year])
    if v > max_volume:
        max_volume = v

fig = go.Figure()

for year in years:
    df_y = df_modern[df_modern['Year'] == year]
    year_volume = len(df_y)
    
    unr_app = len(df_y[df_y['Derived_Status'] == 'Unresolved (At Application)'])
    unr_pc = len(df_y[df_y['Derived_Status'] == 'Unresolved (At PC)'])
    unr_zap = len(df_y[df_y['Derived_Status'] == 'Unresolved (At ZAP)'])
    unr_council = len(df_y[df_y['Derived_Status'] == 'Unresolved (At Council)'])

    unscraped = len(df_y[df_y['Derived_Status'] == 'Approved (Unscraped)'])
    scraped = len(df_y[df_y['Derived_Status'] == 'Approved (Scraped)'])
    total_approved = unscraped + scraped

    pc_council = len(df_y[(df_y['Commission_Type'] == 'PC') & (df_y['Final_Council_Date'].notna())])
    zap_council = len(df_y[(df_y['Commission_Type'] == 'ZAP') & (df_y['Final_Council_Date'].notna())])
    unassigned_council = len(df_y[(df_y['Commission_Type'].isna()) & (df_y['Final_Council_Date'].notna())])

    pc_council += unassigned_council

    pc_total = pc_council + unr_pc
    zap_total = zap_council + unr_zap
    
    dummy_volume = max_volume - year_volume
    
    source = [0, 0, 0, 1, 1, 2, 2, 4, 4, 6]
    target = [3, 1, 2, 3, 4, 3, 4, 5, 3, 7]
    value  = [
        unr_app, pc_total, zap_total,
        unr_pc, pc_council,
        unr_zap, zap_council,
        total_approved, unr_council,
        dummy_volume
    ]
    
    value = [v if v > 0 else 1e-5 for v in value]
    
    link_colors = ["rgba(41, 128, 185, 0.4)"] * (len(source) - 1) + ["rgba(0,0,0,0)"]

    fig.add_trace(go.Sankey(
        visible=False,
        arrangement="fixed",
        valueformat = ",d",
        node = dict(
          pad = 30,
          thickness = 30,
          line = dict(color = ["black"]*6 + ["rgba(0,0,0,0)", "rgba(0,0,0,0)"], width = [0.5]*6 + [0,0]),
          label = labels,
          color = colors,
          x = node_x,
          y = node_y
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          color = link_colors
        ),
        name = str(year)
    ))

if len(fig.data) > 0:
    fig.data[0].visible = True

steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Austin Zoning Pipeline: {years[i]} Volume"}],
        label=str(years[i])
    )
    step["args"][0]["visible"][i] = True
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Filing Year: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    title_text=f"Austin Zoning Pipeline: {years[0]} Volume",
    font_size=16,
    width=1200,
    height=700,
    paper_bgcolor='rgba(255,255,255,1)',
    plot_bgcolor='rgba(255,255,255,1)'
)

fig.write_html(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\interactive_timelapse.html")
print("Raw HTML Generated.")
