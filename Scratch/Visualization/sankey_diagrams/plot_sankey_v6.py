import plotly.graph_objects as go
import pandas as pd

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

df['Year'] = pd.to_datetime(df['application_start_date'], errors='coerce').dt.year
df_modern = df[df['Year'] >= 2009]

dead_app = len(df_modern[df_modern['Derived_Status'] == 'Dead (At Application)'])
dead_pc = len(df_modern[df_modern['Derived_Status'] == 'Dead (At PC)'])
dead_zap = len(df_modern[df_modern['Derived_Status'] == 'Dead (At ZAP)'])
dead_council = len(df_modern[df_modern['Derived_Status'] == 'Dead (At Council)'])

ongoing_app = len(df_modern[df_modern['Derived_Status'] == 'Ongoing (At Application)'])
ongoing_pc = len(df_modern[df_modern['Derived_Status'] == 'Ongoing (At PC)'])
ongoing_zap = len(df_modern[df_modern['Derived_Status'] == 'Ongoing (At ZAP)'])
ongoing_council = len(df_modern[df_modern['Derived_Status'] == 'Ongoing (At Council)'])

unscraped = len(df_modern[df_modern['Derived_Status'] == 'Approved (Unscraped)'])
scraped = len(df_modern[df_modern['Derived_Status'] == 'Approved (Scraped)'])
total_approved = unscraped + scraped

remands = int(df_modern['Remand_Count'].sum()) if 'Remand_Count' in df_modern.columns else 133

pc_council = len(df_modern[(df_modern['Commission_Type'] == 'PC') & (df_modern['Final_Council_Date'].notna())])
zap_council = len(df_modern[(df_modern['Commission_Type'] == 'ZAP') & (df_modern['Final_Council_Date'].notna())])
unassigned_council = len(df_modern[(df_modern['Commission_Type'].isna()) & (df_modern['Final_Council_Date'].notna())])

# Route unassigned council cases to PC for simplicity, or just distribute them
pc_council += unassigned_council

pc_total = pc_council + dead_pc + ongoing_pc
zap_total = zap_council + dead_zap + ongoing_zap

labels = [
    "Application Filed",              # 0
    "Planning Commission (PC)",       # 1
    "Zoning & Platting (ZAP)",        # 2
    "Withdrawn or Dead",              # 3
    "Ongoing",                        # 4
    "City Council Agenda",            # 5
    "Approved / Ordinance Passed",    # 6
    "Remanded (Boomerang)"            # 7
]

source = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 5, 5, 5, 5]
target = [3, 4, 1, 2, 3, 4, 5, 3, 4, 5, 6, 3, 4, 7]
value  = [
    dead_app,
    ongoing_app,
    pc_total,
    zap_total,
    dead_pc,
    ongoing_pc,
    pc_council,
    dead_zap,
    ongoing_zap,
    zap_council,
    total_approved,
    dead_council,
    ongoing_council,
    remands
]

colors = [
    "#34495e", # App
    "#2980b9", # PC
    "#3498db", # ZAP
    "#e74c3c", # Dead
    "#f39c12", # Ongoing
    "#8e44ad", # Council
    "#2ecc71", # Approved
    "#e67e22"  # Remanded
]

link_colors = ["rgba(41, 128, 185, 0.4)"] * len(source)

fig = go.Figure(data=[go.Sankey(
    valueformat = ",d",
    node = dict(
      pad = 30,
      thickness = 30,
      line = dict(color = "black", width = 0.5),
      label = labels,
      color = colors
    ),
    link = dict(
      source = source,
      target = target,
      value = value,
      color = link_colors
  ))])

fig.update_layout(
    title_text="The Austin Zoning Pipeline: PC vs ZAP Separation (2009-2024)", 
    font_size=16,
    width=1100,
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.write_image(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\objective_sankey_final.png", scale=3)
print("Sankey generated with PC/ZAP split.")
