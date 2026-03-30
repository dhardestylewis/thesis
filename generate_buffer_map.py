import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
import os

# Create stylized parcels around 1506 Waller St
fig, ax = plt.subplots(figsize=(8, 8))

# Central Subject Tract (1506 Waller St approximate)
subject_tract = Polygon([[0, 0], [40, 0], [40, 80], [0, 80]], closed=True, 
                        facecolor='#FF9999', edgecolor='red', hatch='//', alpha=0.7, 
                        label="Subject Tract (Case C14-2007-0131)")

# 200ft Buffer
buffer_circle = Circle((20, 40), 200, color='blue', fill=False, linestyle='--', linewidth=2, 
                       label="200ft Statutory Buffer")

ax.add_patch(subject_tract)
ax.add_patch(buffer_circle)

# Protesting Neighbors (Overlapping Buffer)
np.random.seed(42)
protest_parcels = []
for i in range(12):
    # Place protesting parcels overlapping the buffer edge
    angle = np.random.uniform(0, 2*np.pi)
    dist = np.random.uniform(100, 220)
    x = 20 + dist * np.cos(angle)
    y = 40 + dist * np.sin(angle)
    
    # 40x40 parcels
    p = Polygon([[x, y], [x+40, y], [x+40, y+40], [x, y+40]], closed=True,
                facecolor='#99CCFF', edgecolor='blue', alpha=0.6)
    
    if dist < 190: # Only count areas actually inside buffer for visual
        ax.add_patch(p)

    # Note: real petitions calculate EXACT intersecting geometry

# Non-protesting neighbors inside buffer
for i in range(8):
    angle = np.random.uniform(0, 2*np.pi)
    dist = np.random.uniform(80, 180)
    x = 20 + dist * np.cos(angle)
    y = 40 + dist * np.sin(angle)
    
    p = Polygon([[x, y], [x+50, y], [x+50, y+30], [x, y+30]], closed=True,
                facecolor='#EEEEEE', edgecolor='gray', alpha=0.5)
    ax.add_patch(p)

ax.set_xlim(-220, 260)
ax.set_ylim(-180, 300)
ax.set_aspect('equal')
ax.set_title("Visualizing the 200ft Statutory Protest Buffer\n(Demonstrative Reference for Appendix D)", fontsize=14, fontweight='bold')
ax.axis('off')

# Extract custom legend using dummy proxy artists for clear labeling
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(facecolor='#FF9999', edgecolor='red', hatch='//', label='Subject Tract (Zoning Case)')
blue_circle = plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='200ft Statutory Buffer')
protest_patch = mpatches.Patch(facecolor='#99CCFF', edgecolor='blue', label='Valid Protesting Parcel (Area Overlap)')
neutral_patch = mpatches.Patch(facecolor='#EEEEEE', edgecolor='gray', label='Non-Protesting Area')

plt.legend(handles=[red_patch, blue_circle, protest_patch, neutral_patch], loc='lower right', bbox_to_anchor=(1.1, -0.05))

plt.tight_layout()

# Save the figure to the thesis directory
out_path = r'c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Figures\waller_buffer_map.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300)
print(f"Saved {out_path}")
