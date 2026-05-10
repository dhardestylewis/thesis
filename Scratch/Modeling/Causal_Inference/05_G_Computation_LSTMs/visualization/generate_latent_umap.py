import os
import torch
import numpy as np
import pandas as pd
from causal_seq2seq_cvae import Seq2SeqCVAE

try:
    import umap.umap_ as umap
except ImportError:
    print("UMAP not found. Please install: pip install umap-learn")
    import sys; sys.exit(1)
import plotly.express as px

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    PANEL_PATH = r"biweekly_panel.csv"
    if not os.path.exists(PANEL_PATH): PANEL_PATH = r"/data/biweekly_panel.csv"
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    df['vote_friction'] = df['vote_event'] * (1 + df['cumulative_nay_votes'].clip(upper=10))
    df['cumulative_vote_friction'] = df.groupby('case_number')['vote_friction'].cumsum()
    if 'petition_pct_this_period' in df.columns:
        df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
        
    if "pdf_requested_height_ft" in df.columns:
        initial_req = df.groupby("case_number")["pdf_requested_height_ft"].transform("max")
        current_constraint = df[["pdf_requested_height_ft", "pdf_staff_recommends_ht"]].min(axis=1) if "pdf_staff_recommends_ht" in df.columns else df["pdf_requested_height_ft"]
        current_constraint = current_constraint.fillna(initial_req)
        final_ht = df["pdf_reduced_to_ft"].fillna(current_constraint).fillna(0) if "pdf_reduced_to_ft" in df.columns else current_constraint.fillna(0)
        df["net_height_change"] = (initial_req - final_ht).clip(lower=0).fillna(0)
    else:
        df["net_height_change"] = 0

    targets = ["resolved", "cumulative_vote_friction", "net_height_change",
               "council_nlp_total_tokens", "commission_hearings_this_period", "council_hearings_this_period",
               "yea_votes_this_period", "nay_votes_this_period"]

    exclude = set(targets + [
        "case_number", "period_start", "period_start_dt", "year", "quarter", "petition_year", "petition_quarter",
        "latitude", "longitude", "shape_area", "council_district", "census_tract", "land_use_code",
        "label_petition_total_pct", "label_valid_protest", "label_real_days_in_pipeline", 
        "label_valid_petition_pct", "label_exact_geometric_petition_pct",
        "pdf_council_date", "pdf_council_agenda_url", "pdf_council_transcripts", 
        "pdf_commission_date", "pdf_commission_agenda_url", "pdf_commission_transcripts",
        "pdf_requested_zoning", "Final_Zoning", "Initial_Zoning_Base", "Final_Zoning_Base",
        "status_date", "update_date", "status", "ordinance", "T0", "T_vote", "T_end", 
        "censored", "building_age"
    ])

    features = [c for c in df.columns if c not in exclude]
    for f in features + targets:
        if f not in df.columns: df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    norm_dict = {}
    for f in features + ["net_height_change"]:
        if f in ["pdf_requested_height_ft", "council_nlp_total_tokens", "net_vote_margin", "cumulative_yea_votes", "cumulative_nay_votes", "petition_pct_this_period", "cumulative_petition_pct"]:
            continue
        if f in ["land_acres", "market_value", "appraised_value"]:
            df[f] = np.log1p(df[f].clip(lower=0))
        mean_v, std_v = df[f].mean(), df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)

    for f in ["pdf_requested_height_ft", "council_nlp_total_tokens"]:
        df[f] = np.log1p(df[f].clip(lower=0))
    for f in ["commission_hearings_this_period", "council_hearings_this_period"]:
        df[f] = df[f].clip(lower=0, upper=1)

    return df, features, targets, norm_dict

def main():
    print("Loading data for Latent UMAP...")
    df, features, targets, norm_dict = load_data()
    
    ckpt_path = "causal_seq2seq_weights.pt"
    if not os.path.exists(ckpt_path): ckpt_path = "/data/output/causal_seq2seq_weights.pt"
    if not os.path.exists(ckpt_path):
        print(f"Waiting for model weights...")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    
    treat_idx = [features.index("petition_pct_this_period"), features.index("cumulative_petition_pct")]
    confounder_idx = []
    if "proposed_max_far" in features: confounder_idx.append(features.index("proposed_max_far"))
    if "pdf_requested_height_ft" in features: confounder_idx.append(features.index("pdf_requested_height_ft"))
    if "land_acres" in features: confounder_idx.append(features.index("land_acres"))

    model = Seq2SeqCVAE(len(features), treat_idx=treat_idx, confounder_idx=confounder_idx).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    # Get all unique cases
    cases = df['case_number'].unique()
    
    # We only need the very first timestep of each case to encode its structural features!
    # Because structural features are static (or forward-filled), t=1 is perfectly representative of the case structure.
    sub = df.drop_duplicates(subset=['case_number'], keep='first').copy()
    
    for f in ["land_acres", "market_value", "appraised_value"]:
        if f in sub.columns: sub[f] = np.log1p(sub[f].clip(lower=0))
    for f in norm_dict:
        if f in sub.columns: sub[f] = (sub[f] - norm_dict[f][0]) / (norm_dict[f][1] + 1e-8)
    if 'pdf_requested_height_ft' in sub.columns:
        sub['pdf_requested_height_ft'] = np.log1p(sub['pdf_requested_height_ft'].clip(lower=0))
        
    feat_arr = sub[features].values.astype(np.float32)
    
    # The Encoder expects a sequence (batch, seq, num_features). We can just pass a sequence of length 1!
    X = np.expand_dims(feat_arr, axis=1)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        mu, _ = model.encode(X)
        # mu is shape (batch, latent_dim) = (5000+, 32)
        Z = mu.cpu().numpy()
        
    print(f"Projecting {Z.shape[0]} cases from {Z.shape[1]}D down to 3D using UMAP...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(Z)
    
    # Get metadata for plotting
    plot_df = pd.DataFrame({
        'case_number': sub['case_number'],
        'UMAP_1': embedding[:, 0],
        'UMAP_2': embedding[:, 1],
        'UMAP_3': embedding[:, 2],
        'total_delay_periods': sub['label_real_days_in_pipeline'] / 14.0 if 'label_real_days_in_pipeline' in sub.columns else 0,
        'petition_pct': sub['label_petition_total_pct'] if 'label_petition_total_pct' in sub.columns else 0,
        'requested_height': sub['pdf_requested_height_ft'] if 'pdf_requested_height_ft' in sub.columns else 0
    })
    
    print("Generating interactive 3D plot...")
    fig = px.scatter_3d(
        plot_df, x='UMAP_1', y='UMAP_2', z='UMAP_3',
        color='total_delay_periods',
        hover_name='case_number',
        hover_data=['petition_pct', 'requested_height'],
        color_continuous_scale='Inferno_r',
        title='Global Latent Topology of Zoning Vulnerability (UMAP)'
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.write_html("latent_topology_umap.html")
    print("Saved latent_topology_umap.html!")

if __name__ == "__main__":
    main()
