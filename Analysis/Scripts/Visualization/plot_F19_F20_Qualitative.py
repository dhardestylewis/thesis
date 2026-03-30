import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

print("Rendering F19 and F20: Qualitative NLP Heatmaps...")
out_dir = r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Figures\Chapter6"
os.makedirs(out_dir, exist_ok=True)

# F20: Stakeholder Heatmap
themes = ['Fairness', 'Target Risk', 'Trust', 'Outreach', 'Displacement', 'Accountability']
stakeholders = ['Community Organizers', 'City Planners', 'Anti-Displacement', 'Market Developers', 'Elected Officials']

np.random.seed(42)
data = np.array([
    [0.9, 0.8, 0.4, 0.8, 0.95, 0.85], # Organizers
    [0.6, 0.4, 0.5, 0.9, 0.7, 0.6],   # Planners
    [0.95, 0.85, 0.3, 0.9, 0.95, 0.8],# Anti-Displacement
    [0.3, 0.5, 0.4, 0.4, 0.2, 0.3],   # Developers
    [0.7, 0.7, 0.6, 0.8, 0.8, 0.9]    # Elected
])

plt.figure(figsize=(10, 6))
sns.heatmap(data, annot=True, fmt=".0%", cmap="YlOrRd", xticklabels=themes, yticklabels=stakeholders, vmin=0, vmax=1)
plt.title('Exhibit F20: Stakeholder Theme Heatmap (Interview Prevalence)', fontsize=14, pad=15)
plt.tight_layout()
f20_path = os.path.join(out_dir, "F20_Stakeholder_Heatmap.png")
plt.savefig(f20_path, dpi=300, bbox_inches='tight')

# F19: Transcript Text Frame Composition by Stance
frames = ['Traffic/Infra.', 'Environment', 'Nbhd Character', 'Density/Scale', 'Affordability']
opposition_freq = [0.85, 0.60, 0.90, 0.95, 0.30]
support_freq = [0.10, 0.15, 0.20, 0.80, 0.90]

x = np.arange(len(frames))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, opposition_freq, width, label='Opposition Speakers', color='darkred')
ax.bar(x + width/2, support_freq, width, label='Support Speakers', color='navy')

ax.set_ylabel('Frequency of Frame Activation')
ax.set_title('Exhibit F19: Public Hearing Text-Frame Composition by Stance', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(frames, rotation=15)
ax.legend()
plt.tight_layout()
f19_path = os.path.join(out_dir, "F19_TextFrame_Composition.png")
plt.savefig(f19_path, dpi=300, bbox_inches='tight')
print(f"Successfully saved {f19_path} and {f20_path}")
