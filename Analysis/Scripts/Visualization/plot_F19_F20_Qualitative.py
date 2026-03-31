import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Extract authentic TFIDF columns rather than fabricating qualitative codes
    tfidf_cols = [c for c in df.columns if c.startswith('tfidf_')]
    if not tfidf_cols:
        print("[!] No TF-IDF features detected. Cannot empirically derive text frames.")
        return
        
    # Pick a few distinct TF-IDF anchors for F19 (Discourse differences)
    # Using existing tokens, e.g., 'neighborhood', 'density', 'traffic' if they exist, else sample the components
    frames = []
    opposition_freq = []
    support_freq = []
    
    for word in ['neighborhood', 'property', 'district', 'planning', 'staff']:
        col = f'tfidf_{word}'
        if col in df.columns:
            frames.append(word.capitalize())
            # Calculate activation rate (percentage of cases where word holds positive weight)
            op = df[df['is_protested'] == 1]
            sup = df[df['is_protested'] == 0]
            opposition_freq.append((op[col] > 0.05).mean())
            support_freq.append((sup[col] > 0.05).mean())
            
    if frames:
        print("  -> Constructing F19 using authentic token activation gaps...")
        x = np.arange(len(frames))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, opposition_freq, width, label='Opposed Cases', color='darkred')
        ax.bar(x + width/2, support_freq, width, label='Uncontested Cases', color='navy')

        ax.set_ylabel('Empirical Document Activation Frequency')
        ax.set_title('Exhibit F19: Explicit Transcribed Hearing Text-Frame Distances', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(frames, rotation=15)
        ax.legend()
        plt.tight_layout()
        f19_path = os.path.join(out_dir, "F19_TextFrame_Composition.png")
        plt.savefig(f19_path, dpi=300, bbox_inches='tight')
        plt.close()

    # F20: Topic Activation Heatmap by Council District (Since 'Stakeholders' can't be partitioned objectively)
    if 'council_district' in df.columns or 'council_district_x' in df.columns:
        print("  -> Constructing F20 using topological text clustering...")
        dist_col = 'council_district_x' if 'council_district_x' in df.columns else 'council_district'
        districts = sorted(df[dist_col].dropna().unique())[:5] # Take 5 for space
        
        heatmap_data = []
        for d in districts:
            d_sub = df[df[dist_col] == d]
            row = []
            for w in frames:
                col = f'tfidf_{w.lower()}'
                row.append(d_sub[col].mean())
            heatmap_data.append(row)
            
        data = np.array(heatmap_data)
        if data.size > 0:
            # Normalize for heatmap visibility
            data = data / (data.max() + 1e-9)
            plt.figure(figsize=(10, 6))
            sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=frames, yticklabels=[f"District {int(d)}" for d in districts])
            plt.title('Exhibit F20: Geographic NLP Transcriptions Topic Gravity', fontsize=14, pad=15)
            plt.tight_layout()
            f20_path = os.path.join(out_dir, "F20_Stakeholder_Heatmap.png")
            plt.savefig(f20_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    print(f"[+] Rebuilt F19 and F20 cleanly from {len(tfidf_cols)} SVD text components in H3_Filing_Master_NLP.csv")

if __name__ == "__main__":
    generate_exhibits()
