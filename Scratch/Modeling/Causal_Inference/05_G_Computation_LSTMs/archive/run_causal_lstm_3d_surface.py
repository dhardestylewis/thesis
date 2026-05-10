import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
OUT_PLOT = rf"{OUT_DIR}\causal_lstm_3d_surface.png"

class FastHazardLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)

def main():
    print("1. Loading Baseline Dataset for 3D Surface...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", 
        "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", 
        "period_seq", "petition_pct_this_period", "cumulative_petition_pct", "bw_sin", "bw_cos"
    ]
    
    for f in features: df[f] = df[f].fillna(0)
    df["resolved"] = df["resolved"].fillna(0).astype(int)
    
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    max_seq = 15
    groups = df.groupby("case_number")
    X_seq, y_seq = [], []
    
    for _, group in groups:
        seq = group.sort_values("period_seq")[features].values
        target = group.sort_values("period_seq")["resolved"].values
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
    
    K = 20
    X_seq_oversampled = X_seq + [X_seq[i] for i in treated_idx] * K
    y_seq_oversampled = y_seq + [y_seq[i] for i in treated_idx] * K
    
    X_tensor = torch.tensor(np.array(X_seq_oversampled), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq_oversampled), dtype=torch.float32)
    
    print("2. Training Continuous LSTM...")
    model = FastHazardLSTM(len(features))
    pos_weight = torch.tensor([15.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(15):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            mask = (batch_x[:, :, features.index("period_seq")] != 0).float()
            loss = (criterion(logits, batch_y) * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print("\n3. Generating 3D Surface Grid...")
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

    def get_final_survival(tensor):
        with torch.no_grad():
            logits = model(tensor).numpy()[0]
            logits_recalibrated = logits - np.log(K)
            hazards = 1 / (1 + np.exp(-logits_recalibrated))
        return np.prod(1 - hazards)

    vuln_baseline = create_archetype_tensor(8500.0, 0.9)
    
    periods = np.arange(1, 16)
    pcts = np.arange(0, 105, 5) # 0 to 100 in steps of 5
    
    Z = np.zeros((len(pcts), len(periods)))
    
    for i, pct in enumerate(pcts):
        for j, p in enumerate(periods):
            shock_tensor = vuln_baseline.clone()
            # 0-indexed period is p-1
            shock_tensor[0, p-1, features.index("petition_pct_this_period")] = float(pct)
            shock_tensor[0, p-1:, features.index("cumulative_petition_pct")] = float(pct)
            Z[i, j] = get_final_survival(shock_tensor)
            
    print("4. Plotting Interactive 3D Surface with Plotly...")
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Surface(
        z=Z, 
        x=periods, 
        y=pcts,
        colorscale='Magma',
        cmin=0, cmax=1,
        colorbar=dict(title="Survival Probability")
    )])
    
    fig.update_layout(
        title='The "Gravity Well" of the Supermajority Law<br><sup>True Interactive Topographical Map of Survival Across Timing & Intensity</sup>',
        scene=dict(
            xaxis_title='Intervention Timing (Period)',
            yaxis_title='Petition Intensity (%)',
            zaxis_title='Terminal Survival Probability',
            zaxis=dict(range=[0, 1])
        ),
        width=1200,
        height=900,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    OUT_PLOT_HTML = rf"{OUT_DIR}\causal_lstm_true_3d_surface.html"
    fig.write_html(OUT_PLOT_HTML)
    
    print(f"Interactive Master artifact saved to {OUT_PLOT_HTML}")

if __name__ == "__main__":
    main()
