import plotly.graph_objects as go

labels = [
    "1. Application Filed",           # 0
    "2. Commission Hearing",          # 1
    "Killed at Commission",           # 2
    "3. City Council Agenda",         # 3
    "4A. Swift Approval",             # 4
    "3B. Postponement Loop",          # 5
    "Killed at Council",              # 6
    "4B. Compromise Approval",        # 7
    "Remanded (Boomerang)"            # 8
]

source = [0, 1, 1, 3, 3, 5, 5, 5]
target = [1, 2, 3, 4, 5, 6, 7, 8]
value  = [
    3249, # App -> Comm
    1258, # Comm -> Dead
    1991, # Comm -> Council
    1200, # Council -> Swift
    791,  # Council -> Postpone
    50,   # Postpone -> Dead
    650,  # Postpone -> Compromise
    91    # Postpone -> Remand
]

# Define Node Colors
colors = [
    "#34495e", # App
    "#2980b9", # Comm
    "#e74c3c", # Killed Comm
    "#8e44ad", # Council
    "#2ecc71", # Swift
    "#f39c12", # Postpone
    "#c0392b", # Killed Council
    "#27ae60", # Compromise
    "#e67e22"  # Remand
]

# Link colors (transparent versions of the source nodes)
link_colors = [
    "rgba(52, 73, 94, 0.4)",
    "rgba(41, 128, 185, 0.4)",
    "rgba(41, 128, 185, 0.4)",
    "rgba(142, 68, 173, 0.4)",
    "rgba(142, 68, 173, 0.4)",
    "rgba(243, 156, 18, 0.4)",
    "rgba(243, 156, 18, 0.4)",
    "rgba(243, 156, 18, 0.4)"
]

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
    title_text="The Austin Zoning Pipeline: Administrative Attrition and Friction (2009-2024)", 
    font_size=16,
    width=1200,
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.write_image(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\sankey_pipeline.png", scale=3)
print("Sankey diagram generated successfully.")
