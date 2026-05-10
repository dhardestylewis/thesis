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
OUT_PLOT = rf"{OUT_DIR}\causal_lstm_ablation_matrix.png"

class FastHazardLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)

def main():
    print("1. Loading Baseline Dataset for Ablation...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", 
        "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", 
        "period_seq", "petition_event", "cumulative_petition_events", "bw_sin", "bw_cos"
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
        
    treated_idx = [i for i, seq in enumerate(X_seq) if np.max(seq[:, features.index("petition_event")]) > 0]
    
    K = 20
    X_seq_oversampled = X_seq + [X_seq[i] for i in treated_idx] * K
    y_seq_oversampled = y_seq + [y_seq[i] for i in treated_idx] * K
    
    X_tensor = torch.tensor(np.array(X_seq_oversampled), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq_oversampled), dtype=torch.float32)
    
    print("2. Training Baseline LSTM...")
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
            
    print("\n3. Generating 2x2 Causal Ablation Curves...")
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

    def get_survival(tensor):
        with torch.no_grad():
            logits = model(tensor).numpy()[0]
            logits_recalibrated = logits - np.log(K)
            hazards = 1 / (1 + np.exp(-logits_recalibrated))
        return np.cumprod(1 - hazards)

    # The 2x2 Matrix
    sqft_low = 8500.0
    sqft_high = 871200.0
    grav_low = 0.05
    grav_high = 0.90
    
    archetypes = {
        "LowA_LowC": create_archetype_tensor(sqft_low, grav_low),
        "LowA_HighC": create_archetype_tensor(sqft_low, grav_high),
        "HighA_LowC": create_archetype_tensor(sqft_high, grav_low),
        "HighA_HighC": create_archetype_tensor(sqft_high, grav_high),
    }
    
    results = {}
    t_int = 4 # Shock at Period 5
    
    for key, base_tensor in archetypes.items():
        base_surv = get_survival(base_tensor)
        
        shock_tensor = base_tensor.clone()
        shock_tensor[0, t_int, features.index("petition_event")] = 1.0
        shock_tensor[0, t_int:, features.index("cumulative_petition_events")] = 1.0
        shock_surv = get_survival(shock_tensor)
        
        results[key] = {"Baseline": base_surv, "Treated": shock_surv}
        
    print("4. Plotting 2x2 Matrix...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, dpi=300)
    periods = np.arange(1, max_seq + 1)
    
    titles = {
        "LowA_LowC": "Low Acreage, Low Contagion\n(8,500 sqft | 5% Blight)",
        "LowA_HighC": "Low Acreage, High Contagion\n(8,500 sqft | 90% Blight)",
        "HighA_LowC": "High Acreage, Low Contagion\n(870k sqft | 5% Blight)",
        "HighA_HighC": "High Acreage, High Contagion\n(870k sqft | 90% Blight)",
    }
    
    mapping = {
        "LowA_LowC": axes[0,0],
        "HighA_LowC": axes[0,1],
        "LowA_HighC": axes[1,0],
        "HighA_HighC": axes[1,1]
    }
    
    for key, ax in mapping.items():
        surv = results[key]
        ax.plot(periods, surv["Baseline"], color="#3498DB", linewidth=4, label="Control (No Petition)")
        ax.plot(periods, surv["Treated"], color="#E74C3C", linewidth=3, linestyle="--", label="Treated (Petition at Period 5)")
        ax.axvline(5, color="black", linestyle=":", alpha=0.5)
        ax.set_title(titles[key], weight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if key in ["LowA_LowC", "HighA_LowC"]:
            ax.set_xlabel("Biweekly Period")
            
    axes[0,0].set_ylabel("Survival Prob.\n(Low Contagion Row)", fontsize=12, weight="bold")
    axes[1,0].set_ylabel("Survival Prob.\n(High Contagion Row)", fontsize=12, weight="bold")
    axes[1,0].legend(loc='lower left', fontsize=9)
    
    plt.suptitle("Causal Ablation Matrix: Isolating the Driver of Structural Immunity\n(Separating Acreage from Spatial Contagion to identify the true source of developer leverage)", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PLOT, bbox_inches="tight")
    
    print(f"Master artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
