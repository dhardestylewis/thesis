import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

ROOT = r'C:\Users\dhl\data\thesis\thesis'
SEQ_DIR = os.path.join(ROOT, "Thesis_Draft", "Sequential_Clustermaps")

anchors = [2018, 2019, 2020, 2021, 2022, 2023]

print("[*] Retrieving Canonical Architecture & Feature indexing explicitly...")
# We use the 2018 array completely as the Canonical structural ordering identically across all maps.
# Better yet, compute a master sum to find the canonical hierarchy of features
master_df = None
for a in anchors:
    csv_path = os.path.join(SEQ_DIR, f"LTR_Clustermap_Matrix_{a}.csv")
    if not os.path.exists(csv_path): continue
    d = pd.read_csv(csv_path, index_col=0)
    if master_df is None:
        master_df = d
    else:
        master_df = master_df.add(d, fill_value=0)

# Establish strict structural sorting
feature_order = master_df.sum(axis=1).sort_values(ascending=False).index.tolist()
arch_order = master_df.sum(axis=0).sort_values(ascending=False).index.tolist() # Or keep Base vs Meta strictly grouped!
arch_order = sorted(arch_order, key=lambda x: 'Meta' in x) # Forces Base Models left, Meta Models right strictly!

png_files = []

print("[*] Re-rendering exactly sequenced spatial Heatmaps organically...")
sns.set_theme(style="white", context="paper", font_scale=1.0)

for anchor in anchors:
    csv_path = os.path.join(SEQ_DIR, f"LTR_Clustermap_Matrix_{a}.csv") # bug: should be '{anchor}' Wait, fixed below!
    csv_path = os.path.join(SEQ_DIR, f"LTR_Clustermap_Matrix_{anchor}.csv")
    if not os.path.exists(csv_path): continue
    
    df = pd.read_csv(csv_path, index_col=0)
    
    # Reindex identically explicitly filling missing features with 0.0
    df = df.reindex(index=feature_order, columns=arch_order, fill_value=0.0)
    
    plt.figure(figsize=(14, 12))
    # We use a standard heatmap instead of clustermap to rigidly enforce the canonical cell indexing across sequences
    ax = sns.heatmap(
        df, 
        cmap="crest", 
        linewidths=.5, 
        annot=False, 
        cbar_kws={'label': 'NDCG-Weighted Relational Importance (%)'},
        vmin=0.0, vmax=20.0 # Force uniform color intensity scaling across the GIF structurally
    )
    plt.title(f"Sequential Chronology: Anchor {anchor}\nObserving Explicit Topographical Drift and Meta-Poisoning Natively", 
                    fontsize=16, weight='bold', pad=20)
    plt.ylabel("Invariant Geographic Features (Sorted by Macro Absolute)", fontsize=14)
    plt.xlabel("Algorithmic Limit Configurations (Base Topologies -> Meta Topologies)", fontsize=14)
    plt.tight_layout()
    
    out_png = os.path.join(SEQ_DIR, f"plot_fixed_heatmap_{anchor}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    png_files.append(out_png)
    print(f"--> Overrode structural mapping beautifully for {anchor}")

print("[*] Stitching longitudinal matrices natively into infinite GIF...")
if len(png_files) > 0:
    images = []
    for f in png_files:
        images.append(Image.open(f))
    
    gif_path = os.path.join(SEQ_DIR, "plot_ltr_clustermap_timelapse.gif")
    
    # Save GIF natively natively appending duplicates of 2023 at the end to hold the final frame logically
    final_frame = images[-1]
    export_images = images + [final_frame] * 3 
    
    images[0].save(
        gif_path,
        save_all=True,
        append_images=export_images[1:],
        duration=1500, # 1.5 Seconds per sequence frame smoothly
        loop=0 # Infinite Loop strictly
    )
    print(f"Dumped Master Time-lapse Array securely to: {gif_path}")
else:
    print("Failed to locate matrices.")

