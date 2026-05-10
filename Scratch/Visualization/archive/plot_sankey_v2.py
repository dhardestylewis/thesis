import plotly.graph_objects as go
import pandas as pd
import re

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

df['Year'] = pd.to_datetime(df['application_start_date'], errors='coerce').dt.year
df_modern = df[df['Year'] >= 2009]

total_applied = len(df_modern)
dead = len(df_modern[df_modern['Derived_Status'] == 'Withdrawn_or_Dead'])
ongoing = len(df_modern[df_modern['Derived_Status'] == 'Ongoing'])
unscraped = len(df_modern[df_modern['Derived_Status'] == 'Completed (Unscraped)'])

scraped = df_modern[df_modern['Derived_Status'] == 'Completed (Scraped)']
scraped_total = len(scraped)
postponed = len(scraped[scraped['Council_Appearances'] > 1])
swift = scraped_total - postponed

# We don't have exact remand numbers without deeper NLP, so we omit boomerang for simplicity here 
# or just route a small % of postponements to compromise vs swift vs dead based on general heuristics,
# but the user wanted exact numbers so let's stick to the exact data we have.

labels = [
    "1. Application Filed",           # 0
    "2. Commission / Review",         # 1
    "Killed / Withdrawn (Dead)",      # 2
    "Ongoing / In Review",            # 3
    "Approved (Unscraped)",           # 4
    "3. City Council Agenda",         # 5
    "4A. Swift Approval",             # 6
    "3B. Postponement Loop",          # 7
    "4B. Delayed Approval"            # 8
]

source = [0, 1, 1, 1, 1, 5, 5, 7]
target = [1, 2, 3, 4, 5, 6, 7, 8]
value  = [
    total_applied, # App -> Comm
    dead,          # Comm -> Dead
    ongoing,       # Comm -> Ongoing
    unscraped,     # Comm -> Unscraped
    scraped_total, # Comm -> Council
    swift,         # Council -> Swift
    postponed,     # Council -> Postpone
    postponed      # Postpone -> Delayed Approval
]

colors = [
    "#34495e", # App
    "#2980b9", # Comm
    "#e74c3c", # Dead
    "#f39c12", # Ongoing
    "#27ae60", # Unscraped
    "#8e44ad", # Council
    "#2ecc71", # Swift
    "#f39c12", # Postpone
    "#27ae60"  # Delayed
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
    title_text="The Austin Zoning Pipeline: Corrected Case Attrition (2009-2024)", 
    font_size=14,
    width=1200,
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.write_image(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\sankey_pipeline.png", scale=3)
print("Updated Sankey diagram generated successfully.")
