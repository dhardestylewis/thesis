import plotly.graph_objects as go
import pandas as pd

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

df['Year'] = pd.to_datetime(df['application_start_date'], errors='coerce').dt.year
df_modern = df[df['Year'] >= 2009]

total_applied = len(df_modern)
dead = len(df_modern[df_modern['Derived_Status'] == 'Withdrawn_or_Dead'])
ongoing = len(df_modern[df_modern['Derived_Status'] == 'Ongoing'])
unscraped = len(df_modern[df_modern['Derived_Status'] == 'Completed (Unscraped)'])
scraped = len(df_modern[df_modern['Derived_Status'] == 'Completed (Scraped)'])

total_approved = unscraped + scraped
remands = 133
council_total = total_approved + remands

labels = [
    "Application Filed",              # 0
    "Administrative Review",          # 1
    "Withdrawn or Dead",              # 2
    "Ongoing",                        # 3
    "City Council Agenda",            # 4
    "Approved / Ordinance Passed",    # 5
    "Remanded (Boomerang)"            # 6
]

source = [0, 1, 1, 1, 4, 4]
target = [1, 2, 3, 4, 5, 6]
value  = [
    total_applied,  # App -> Comm
    dead,           # Comm -> Dead
    ongoing,        # Comm -> Ongoing
    council_total,  # Comm -> Council
    total_approved, # Council -> Approved
    remands         # Council -> Remanded
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
    title_text="The Austin Zoning Pipeline: Objective Administrative Outcomes (2009-2024)", 
    font_size=16,
    width=1100,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.write_image(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\objective_sankey.png", scale=3)
print("Objective sankey generated.")
