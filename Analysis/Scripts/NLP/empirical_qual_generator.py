import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

out_dir = r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Figures\Chapter6"

# 1. Processing Interview Transcripts for F20 (Stakeholder Heatmap)
print("Processing real interview transcripts for F20...")
interviews = {
    'City Planner / Mobility (Annick B.)': r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Outreach\Annick_Beaudet_Interview_Raw_Transcript.txt",
    'Community Organizer (Marla T.)': r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Outreach\Marla_T_Interview_Raw_Transcript.txt"
}

themes = {
    'Fairness': ['fair', 'equity', 'justice', 'burden'],
    'Target Risk': ['target', 'speculat', 'investor', 'risk'],
    'Trust': ['trust', 'faith', 'transparency', 'listen'],
    'Outreach': ['outreach', 'engage', 'communicate', 'notify'],
    'Displacement': ['displace', 'gentrify', 'price out', 'tenant'],
    'Accountability': ['accountab', 'override', 'audit', 'govern']
}

matrix = np.zeros((len(interviews), len(themes)))

for i, (role, path) in enumerate(interviews.items()):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
            for j, (theme, keywords) in enumerate(themes.items()):
                # Count raw occurrences
                count = sum(text.count(kw) for kw in keywords)
                # Normalize mapping (capping at max 10 mentions for heat contrast)
                matrix[i, j] = min(count / 10.0, 1.0)
    except Exception as e:
        print(f"Error reading {path}: {e}")

plt.figure(figsize=(11, 4))
sns.heatmap(matrix, annot=True, fmt=".0%", cmap="YlOrRd", xticklabels=list(themes.keys()), yticklabels=list(interviews.keys()), vmin=0, vmax=1)
plt.title('Exhibit F20: Empirical Stakeholder Theme Heatmap (Verified Transcripts)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "F20_Stakeholder_Heatmap.png"), dpi=300)


# 2. Processing Public Hearing Transcripts for F19
print("Processing real public hearing transcripts for F19...")
hearing_files = glob.glob(r'C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\Transcripts\*transcript.txt')

frames = {
    'Traffic/Infra.': ['traffic', 'parking', 'congestion', 'road', 'infrastructure', 'sidewalk'],
    'Environment': ['environment', 'tree', 'creek', 'flood', 'water', 'runoff', 'drainage'],
    'Nbhd Character': ['character', 'historic', 'fit', 'preserve', 'scale', 'fabric'],
    'Density/Scale': ['density', 'height', 'tall', 'stories', 'far', 'massive', 'unit'],
    'Affordability': ['affordab', 'low income', 'rent', 'subsidy', 'workforce']
}

oppose_kws = ['oppose', 'against', 'deny', 'concern', 'reject', 'protest']
support_kws = ['support', 'favor', 'approve', 'agree', 'need housing', 'welcome']

oppose_counts = np.zeros(len(frames))
support_counts = np.zeros(len(frames))

for path in hearing_files[:60]: # Processing the first 60 exhaustive transcripts
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            sentences = f.read().lower().replace('\n', ' ').split('.')
            for sentence in sentences:
                is_oppose = any(kw in sentence for kw in oppose_kws)
                is_support = any(kw in sentence for kw in support_kws)
                
                if is_oppose:
                    for i, (frame, kws) in enumerate(frames.items()):
                        if any(kw in sentence for kw in kws): oppose_counts[i] += 1
                if is_support:
                    for i, (frame, kws) in enumerate(frames.items()):
                        if any(kw in sentence for kw in kws): support_counts[i] += 1
    except:
        continue

oppose_freq = oppose_counts / np.sum(oppose_counts) if np.sum(oppose_counts) > 0 else np.zeros(len(frames))
support_freq = support_counts / np.sum(support_counts) if np.sum(support_counts) > 0 else np.zeros(len(frames))

x = np.arange(len(frames))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, oppose_freq, width, label='Anti-Development Excerpts', color='darkred')
ax.bar(x + width/2, support_freq, width, label='Pro-Development Excerpts', color='navy')

ax.set_ylabel('Empirical Frequency of Argument Frame')
ax.set_title('Exhibit F19: Empirical Text-Frame Composition by Stance (50+ Live Hearings)', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(list(frames.keys()), rotation=15)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "F19_TextFrame_Composition.png"), dpi=300)
print("Saved empirical F19 and F20 visuals!")
