import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

import pandas as pd
import os

def generate_exhibits():
    print("[*] Rendering Empirical NLP Aggregations (F19, F20)...")
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    NLP_DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H3_Filing_Master_NLP.csv")
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter6")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(NLP_DATA):
        print(f"[!] F19/F20 Failure: Requires H3_Filing_Master_NLP.csv")
        return

    df = pd.read_csv(NLP_DATA, low_memory=False)
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)
    
    # Extract explicit frame probabilities generated via the Active Learning LLM pipeline
    frame_cols = [c for c in df.columns if c.startswith('prob_frame_')]
    if not frame_cols:
        print("[!] No probability features detected. Cannot empirically derive text frames. Run 09b_nlp_active_learning.py + 30_enrich_h3_nlp.py first.")
        return
        
    frames = []
    opposition_freq = []
    support_freq = []
    
    for col in frame_cols:
        word = col.replace('prob_frame_', '').capitalize()
        # Ensure name matches expectations
        frames.append(word)
        # Calculate activation rate (expected probability)
        op = df[df['is_protested'] == 1]
        sup = df[df['is_protested'] == 0]
        opposition_freq.append(op[col].mean())
        support_freq.append(sup[col].mean())
            
    if frames:
        print("  -> Constructing F19 using authentic qualitative probabilities...")
        x = np.arange(len(frames))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, opposition_freq, width, label='Opposed Cases', color='darkred')
        ax.bar(x + width/2, support_freq, width, label='Uncontested Cases', color='navy')

        ax.set_ylabel('Mean Predicted Frame Probability')
        ax.set_title('Exhibit F19: Transcribed Hearing Text-Frame Composition (Active Learning)', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(frames, rotation=15)
        ax.legend()
        plt.tight_layout()
        f19_path = os.path.join(out_dir, "F19_TextFrame_Composition.png")
        plt.savefig(f19_path, dpi=300, bbox_inches='tight')
        plt.close()

    # F20: Topic Activation Spatial Choropleth by Council District
    if 'council_district' in df.columns or 'council_district_x' in df.columns:
        print("  -> Constructing F19/F20 Spatial Multiples using qualitative active learning clustering...")
        import geopandas as gpd
        
        dist_col = 'council_district_x' if 'council_district_x' in df.columns else 'council_district'
        districts = sorted(df[dist_col].dropna().unique())
        
        # Aggregate mean probabilities
        heatmap_data = []
        for d in districts:
            d_sub = df[df[dist_col] == d]
            row = {'COUNCIL_DI': int(d)}
            for col in frame_cols:
                row[col] = d_sub[col].mean()
            heatmap_data.append(row)
            
        df_agg = pd.DataFrame(heatmap_data)
        
        # Load Geometries
        geojson_path = os.path.join(ROOT, "Data", "GIS", "council_districts.geojson")
        if os.path.exists(geojson_path):
            gdf = gpd.read_file(geojson_path)
            gdf['COUNCIL_DI'] = gdf['COUNCIL_DI'].astype(int)
            
            # Merge text probabilities into geometries
            gdf_merged = gdf.merge(df_agg, on='COUNCIL_DI', how='inner')
            
            # Determine global upper bound for colorbar normalization
            global_max = df_agg[frame_cols].values.max() + 1e-9
            
            n_frames = len(frame_cols)
            fig, axes = plt.subplots(1, n_frames, figsize=(3.5 * n_frames, 5))
            if n_frames == 1:
                axes = [axes]
                
            for i, (col, frame_name) in enumerate(zip(frame_cols, frames)):
                ax = axes[i]
                # Normalize values to global maximum cell value for pure visual comparison
                gdf_merged['norm_' + col] = gdf_merged[col] / global_max
                
                gdf_merged.plot(
                    column='norm_' + col, 
                    cmap="YlOrRd", 
                    ax=ax, 
                    edgecolor='black', 
                    linewidth=0.5, 
                    vmin=0, 
                    vmax=1
                )
                ax.set_title(frame_name, fontsize=12)
                ax.axis('off')
                
            plt.suptitle('Figure 19: Argument Frame Prevalence by Council District', fontsize=16, y=1.02)
            plt.tight_layout()
            f20_path = os.path.join(out_dir, "F20_Stakeholder_Spatial_Multiples.png")
            plt.savefig(f20_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("[+] Successfully rendered Spatial Multiples into F20_Stakeholder_Spatial_Multiples.png")
        else:
            print("[!] Could not locate council_districts.geojson for spatial F19/F20.")

if __name__ == "__main__":
    generate_exhibits()
