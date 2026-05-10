import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
OUT_PLOT = rf"{OUT_DIR}\causal_lstm_multi_intervention_recalibrated.png"

# PyTorch LSTM Architecture
class CausalHazardLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Output is hazard probability (0 to 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        hazard = self.fc(lstm_out)
        return hazard.squeeze(-1)

def main():
    print("1. Loading and Prepping Biweekly Panel for PyTorch...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", 
        "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", 
        "period_seq", "petition_event", "cumulative_petition_events", 
        "bw_sin", "bw_cos"
    ]
    
    # Fill NAs
    for f in features:
        df[f] = df[f].fillna(0)
        
    df["resolved"] = df["resolved"].fillna(0).astype(int)
    
    # Normalize Continuous Features to help LSTM converge
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
    
    # Pack sequences (Truncate to 15 periods for simulation stability)
    max_seq = 15
    groups = df.groupby("case_number")
    X_seq, y_seq = [], []
    
    print("   > Packing Tensors...")
    for _, group in groups:
        seq = group.sort_values("period_seq")[features].values
        target = group.sort_values("period_seq")["resolved"].values
        
        if len(seq) > max_seq:
            seq = seq[:max_seq]
            target = target[:max_seq]
            
        # Pad if shorter
        if len(seq) < max_seq:
            pad_len = max_seq - len(seq)
            pad_x = np.zeros((pad_len, len(features)))
            pad_y = np.zeros(pad_len)
            seq = np.vstack([seq, pad_x])
            target = np.concatenate([target, pad_y])
            
        X_seq.append(seq)
        y_seq.append(target)
        
    # Oversample treated sequences to force LSTM to learn the rare treatment effect
    treated_idx = [i for i, seq in enumerate(X_seq) if np.max(seq[:, features.index('petition_event')]) > 0]
    
    # Duplicate treated sequences 20x
    X_seq_oversampled = X_seq + [X_seq[i] for i in treated_idx] * 20
    y_seq_oversampled = y_seq + [y_seq[i] for i in treated_idx] * 20
    
    X_tensor = torch.tensor(np.array(X_seq_oversampled), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq_oversampled), dtype=torch.float32)
    
    print("2. Training Causal LSTM Hazard Network...")
    model = CausalHazardLSTM(input_dim=len(features), hidden_dim=64)
    
    # Extreme class imbalance: 'resolved' is 1 only once per case, and 0 for 14 periods.
    # We must use BCEWithLogitsLoss to apply a pos_weight
    # The output of the model currently has Sigmoid, so we must remove it from the architecture
    
    # Redefine model dynamically here to remove Sigmoid
    class FastHazardLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(features), 64, 1, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out).squeeze(-1)
            
    model = FastHazardLSTM()
    
    # Calculate pos_weight
    pos_weight = torch.tensor([15.0]) # Roughly 15:1 ratio of 0s to 1s
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(15): # Increased epochs for convergence
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            
            mask = (batch_x[:, :, features.index("period_seq")] != 0).float()
            loss = (criterion(logits, batch_y) * mask).sum() / mask.sum()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"   > Epoch {epoch} Loss: {total_loss/len(loader):.4f}")
        
    print("\n3. Generating G-Computation Counterfactuals (Multi-Intervention Sweep)...")
    model.eval()
    
    def create_archetype_tensor(acres, spatial_grav):
        t = np.zeros((1, max_seq, len(features)))
        for i in range(max_seq):
            t[0, i, features.index("land_acres")] = (acres - norm_dict["land_acres"][0]) / norm_dict["land_acres"][1]
            t[0, i, features.index("archetype_pct_Spatial_Gravity")] = (spatial_grav - norm_dict["archetype_pct_Spatial_Gravity"][0]) / norm_dict["archetype_pct_Spatial_Gravity"][1]
            t[0, i, features.index("period_seq")] = ((i+1) - norm_dict["period_seq"][0]) / norm_dict["period_seq"][1]
            t[0, i, features.index("local_unemployment_rate")] = 0 
            t[0, i, features.index("mortgage_rate_30yr")] = 0
        return torch.tensor(t, dtype=torch.float32)

    # Vulnerable: 0.2 acres, 90% Spatial Blight
    vuln_baseline = create_archetype_tensor(0.2, 0.9)
    # Invincible: 150 acres, 5% Spatial Blight
    inv_baseline = create_archetype_tensor(150.0, 0.05)
    
    intervention_periods = [2, 5, 8, 11] # Zero-indexed for tensors
    
    vuln_curves = {}
    inv_curves = {}
    
    def get_survival(tensor):
        with torch.no_grad():
            logits = model(tensor).numpy()[0]
            # Posterior Recalibration: We oversampled the treated class by K=20.
            # To recover true probabilities, we adjust the log-odds: Logit_true = Logit_pred - ln(K)
            logits_recalibrated = logits - np.log(20)
            hazards = 1 / (1 + np.exp(-logits_recalibrated)) # Sigmoid
        return np.cumprod(1 - hazards)
        
    # Baseline curves
    vuln_curves["Baseline"] = get_survival(vuln_baseline)
    inv_curves["Baseline"] = get_survival(inv_baseline)
    
    for t_int in intervention_periods:
        v_shock = vuln_baseline.clone()
        v_shock[0, t_int, features.index("petition_event")] = 1.0
        v_shock[0, t_int:, features.index("cumulative_petition_events")] = 1.0
        vuln_curves[t_int] = get_survival(v_shock)
        
        i_shock = inv_baseline.clone()
        i_shock[0, t_int, features.index("petition_event")] = 1.0
        i_shock[0, t_int:, features.index("cumulative_petition_events")] = 1.0
        inv_curves[t_int] = get_survival(i_shock)
    
    print("4. Plotting Multi-Intervention Divergence...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    periods = np.arange(1, max_seq + 1)
    
    colors = ["#F1C40F", "#E67E22", "#E74C3C", "#8E44AD"] # Gradient from early to late
    
    # Plot Low-Acreage
    axes[0].plot(periods, vuln_curves["Baseline"], color="#3498DB", linewidth=4, label="Control Trajectory")
    for idx, t_int in enumerate(intervention_periods):
        axes[0].plot(periods, vuln_curves[t_int], color=colors[idx], linewidth=2.5, linestyle="--", label=f"Treated (Period {t_int+1})")
        axes[0].scatter([t_int+1], [vuln_curves["Baseline"][t_int]], color=colors[idx], s=100, zorder=5)
        
    axes[0].set_title("Low-Acreage, High-Contagion Parcel\n(0.2 Acres | 90% Spatial Contagion)", weight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Estimated Survival Probability")
    axes[0].set_xlabel("Biweekly Period")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='lower left', fontsize=9)
    
    # Plot High-Acreage
    axes[1].plot(periods, inv_curves["Baseline"], color="#3498DB", linewidth=4, label="Control Trajectory")
    for idx, t_int in enumerate(intervention_periods):
        axes[1].plot(periods, inv_curves[t_int], color=colors[idx], linewidth=2.5, linestyle="--", label=f"Treated (Period {t_int+1})")
        axes[1].scatter([t_int+1], [inv_curves["Baseline"][t_int]], color=colors[idx], s=100, zorder=5)
        
    axes[1].set_title("High-Acreage, Low-Contagion Parcel\n(150 Acres | 5% Spatial Contagion)", weight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Biweekly Period")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='lower left', fontsize=9)
    
    plt.suptitle("Causal LSTM Multi-Intervention Sweep: Critical Windows of Vulnerability\n(Simulating the temporal treatment effect across various intervention stages)", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PLOT, bbox_inches="tight")
    
    print(f"Master artifact saved to {OUT_PLOT}")
    
    print(f"Master artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
