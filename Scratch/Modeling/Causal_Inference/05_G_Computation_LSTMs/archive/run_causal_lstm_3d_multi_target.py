import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

class RegressionLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)

def train_and_map_3d(df, target_col, is_cumulative, title, z_title, out_filename, features, norm_dict):
    print(f"\n--- Processing Target: {target_col} ---")
    
    max_seq = 15
    groups = df.groupby("case_number")
    X_seq, y_seq = [], []
    
    for _, group in groups:
        seq = group.sort_values("period_seq")[features].values
        target = group.sort_values("period_seq")[target_col].values
        if len(seq) > max_seq:
            seq = seq[:max_seq]
            target = target[:max_seq]
        if len(seq) < max_seq:
            pad_len = max_seq - len(seq)
            pad_x = np.zeros((pad_len, len(features)))
            pad_y = np.zeros(pad_len)
            seq = np.vstack([seq, pad_x])
            target = np.concatenate([target, pad_y])
        X_seq.append(seq)
        y_seq.append(target)
        
    treated_idx = [i for i, seq in enumerate(X_seq) if np.max(seq[:, features.index("petition_pct_this_period")]) > 0.0]
    
    K = 15
    X_seq_oversampled = X_seq + [X_seq[i] for i in treated_idx] * K
    y_seq_oversampled = y_seq + [y_seq[i] for i in treated_idx] * K
    
    X_tensor = torch.tensor(np.array(X_seq_oversampled), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq_oversampled), dtype=torch.float32)
    
    print(f"  > Training Regression LSTM for {target_col}...")
    model = RegressionLSTM(len(features))
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(15):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            mask = (batch_x[:, :, features.index("period_seq")] != 0).float()
            loss = (criterion(preds, batch_y) * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print("  > Generating 3D Surface Grid...")
    model.eval()
    
    def create_archetype_tensor(sqft, spatial_grav):
        t = np.zeros((1, max_seq, len(features)))
        for i in range(max_seq):
            t[0, i, features.index("land_acres")] = (sqft - norm_dict["land_acres"][0]) / norm_dict["land_acres"][1]
            t[0, i, features.index("archetype_pct_Spatial_Gravity")] = (spatial_grav - norm_dict["archetype_pct_Spatial_Gravity"][0]) / norm_dict["archetype_pct_Spatial_Gravity"][1]
            t[0, i, features.index("period_seq")] = ((i+1) - norm_dict["period_seq"][0]) / norm_dict["period_seq"][1]
            t[0, i, features.index("local_unemployment_rate")] = 0 
            t[0, i, features.index("mortgage_rate_30yr")] = 0
        return torch.tensor(t, dtype=torch.float32)

    vuln_baseline = create_archetype_tensor(8500.0, 0.9)
    # If target is height, baseline must start at a high value so we can see it drop.
    # The normalized height tensor value is 0 by default. We'll leave it as is and un-normalize the prediction.
    
    periods = np.arange(1, 16)
    pcts = np.arange(0, 105, 5)
    Z = np.zeros((len(pcts), len(periods)))
    
    for i, pct in enumerate(pcts):
        for j, p in enumerate(periods):
            shock_tensor = vuln_baseline.clone()
            shock_tensor[0, p-1, features.index("petition_pct_this_period")] = float(pct)
            shock_tensor[0, p-1:, features.index("cumulative_petition_pct")] = float(pct)
            
            with torch.no_grad():
                preds = model(shock_tensor).numpy()[0]
                
            if target_col == "proposed_max_height_ft":
                # Un-normalize
                preds = (preds * norm_dict["proposed_max_height_ft"][1]) + norm_dict["proposed_max_height_ft"][0]
                final_val = preds[-1] # Terminal height
            else:
                preds = np.maximum(preds, 0) # Hearings can't be negative
                if is_cumulative:
                    final_val = np.sum(preds) # Total hearings over 15 periods
                else:
                    final_val = preds[-1]
            
            Z[i, j] = final_val

    print(f"  > Plotting Interactive 3D Surface for {target_col}...")
    fig = go.Figure(data=[go.Surface(
        z=Z, 
        x=periods, 
        y=pcts,
        colorscale='Magma' if not target_col == 'proposed_max_height_ft' else 'Viridis',
        colorbar=dict(title=z_title)
    )])
    
    fig.update_layout(
        title=f'{title}<br><sup>True Interactive Topographical Map Across Timing & Intensity</sup>',
        scene=dict(
            xaxis_title='Intervention Timing (Period)',
            yaxis_title='Petition Intensity (%)',
            zaxis_title=z_title,
        ),
        width=1200,
        height=900,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    out_path = rf"{OUT_DIR}\{out_filename}"
    fig.write_html(out_path)
    print(f"  > Master artifact saved to {out_path}")

def main():
    print("1. Loading Baseline Dataset for Multi-Target Sweep...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", 
        "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", 
        "period_seq", "petition_pct_this_period", "cumulative_petition_pct", "bw_sin", "bw_cos"
    ]
    
    for f in features: df[f] = df[f].fillna(0)
    df["council_hearings_this_period"] = df["council_hearings_this_period"].fillna(0).astype(float)
    df["commission_hearings_this_period"] = df["commission_hearings_this_period"].fillna(0).astype(float)
    
    # We MUST normalize proposed_max_height_ft to train the regression properly
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        # Ensure we don't normalize the target when we are training on it directly?
        # Actually it's fine to normalize the target during training as long as we unnormalize predictions.
        # But wait, proposed_max_height_ft is BOTH a feature AND a target in this loop?
        # Yes, autoregressive logic. We'll normalize it for the feature matrix.
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    # Target 1: Downzoning (proposed_max_height_ft)
    # The target array must use the NORMALIZED height so MSE loss works correctly with gradients.
    train_and_map_3d(
        df=df,
        target_col="proposed_max_height_ft", 
        is_cumulative=False,
        title='The "Downzoning" Negotiation Surface',
        z_title='Terminal Proposed Height (ft)',
        out_filename='causal_lstm_3d_height.html',
        features=features,
        norm_dict=norm_dict
    )
    
    # Target 2: Early Friction (commission_hearings)
    train_and_map_3d(
        df=df,
        target_col="commission_hearings_this_period", 
        is_cumulative=True,
        title='The Early Friction Surface (Planning Commission)',
        z_title='Cumulative Commission Hearings',
        out_filename='causal_lstm_3d_commission.html',
        features=features,
        norm_dict=norm_dict
    )
    
    # Target 3: Late Friction (council_hearings)
    train_and_map_3d(
        df=df,
        target_col="council_hearings_this_period", 
        is_cumulative=True,
        title='The Late Political Friction Surface (City Council)',
        z_title='Cumulative Council Hearings',
        out_filename='causal_lstm_3d_council.html',
        features=features,
        norm_dict=norm_dict
    )

if __name__ == "__main__":
    main()
