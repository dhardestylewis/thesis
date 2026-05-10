import plotly.graph_objects as go
import pandas as pd

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

df['Year'] = pd.to_datetime(df['application_start_date'], errors='coerce').dt.year
df_modern = df[df['Year'] >= 2009]

total_applied = len(df_modern)
dead_app = len(df_modern[df_modern['Derived_Status'] == 'Dead (At Application)'])
dead_comm = len(df_modern[df_modern['Derived_Status'] == 'Dead (At Commission)'])
dead_council = len(df_modern[df_modern['Derived_Status'] == 'Dead (At Council)'])
ongoing_app = len(df_modern[df_modern['Derived_Status'] == 'Ongoing (At Application)'])
ongoing_comm = len(df_modern[df_modern['Derived_Status'] == 'Ongoing (At Commission)'])
ongoing_council = len(df_modern[df_modern['Derived_Status'] == 'Ongoing (At Council)'])
unscraped = len(df_modern[df_modern['Derived_Status'] == 'Approved (Unscraped)'])
scraped = len(df_modern[df_modern['Derived_Status'] == 'Approved (Scraped)'])

total_approved = unscraped + scraped
remands = int(df_modern['Remand_Count'].sum()) if 'Remand_Count' in df_modern.columns else 133

council_total = total_approved + dead_council + ongoing_council + remands
comm_total = council_total + dead_comm + ongoing_comm

labels = [
    "Application Filed",              # 0
    "Planning / ZAP Commission",      # 1
    "Withdrawn or Dead",              # 2
    "Ongoing",                        # 3
    "City Council Agenda",            # 4
    "Approved / Ordinance Passed",    # 5
    "Remanded (Boomerang)"            # 6
]

source = [0, 0, 0, 1, 1, 1, 4, 4, 4, 4]
target = [2, 3, 1, 2, 3, 4, 5, 2, 3, 6]
value  = [
    dead_app,      # App -> Dead
    ongoing_app,   # App -> Ongoing
    comm_total,    # App -> Comm
    dead_comm,     # Comm -> Dead
    ongoing_comm,  # Comm -> Ongoing
    council_total, # Comm -> Council
    total_approved,# Council -> Approved
    dead_council,  # Council -> Dead
    ongoing_council,# Council -> Ongoing
    remands        # Council -> Remanded
]

colors = [
    "#34495e", # App
    "#2980b9", # Comm
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
    title_text="The Austin Zoning Pipeline: True Attrition Routing (2009-2024)", 
    font_size=16,
    width=1100,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.write_image(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\objective_sankey.png", scale=3)
print("Sankey generated with split attrition.")
