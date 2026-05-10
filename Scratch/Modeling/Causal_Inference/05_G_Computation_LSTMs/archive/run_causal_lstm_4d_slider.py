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

class UnifiedLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)

def train_model(df, target_col, features, task_type):
    print(f"  > Training Unified LSTM for {target_col} (Max Horizon = 30)...")
    max_seq = 30
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
    K = 15 if task_type == "regression" else 20
    
    X_seq_oversampled = X_seq + [X_seq[i] for i in treated_idx] * K
    y_seq_oversampled = y_seq + [y_seq[i] for i in treated_idx] * K
    
    X_tensor = torch.tensor(np.array(X_seq_oversampled), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq_oversampled), dtype=torch.float32)
    
    model = UnifiedLSTM(len(features))
    if task_type == "classification":
        pos_weight = torch.tensor([15.0])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    else:
        criterion = nn.MSELoss(reduction='none')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(15):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            mask = (batch_x[:, :, features.index("period_seq")] != 0).float()
            loss = (criterion(preds, batch_y) * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model, K

def map_4d_surface(model, survival_model, K_surv, df, target_col, is_cumulative, task_type, title, z_title, out_filename, features, norm_dict):
    print(f"\n--- Generating 4D Slider for Target: {target_col} ---")
    
    def create_archetype_tensor(sqft, spatial_grav, horizon):
        t = np.zeros((1, 30, len(features)))
        for i in range(horizon):
            t[0, i, features.index("land_acres")] = (sqft - norm_dict["land_acres"][0]) / norm_dict["land_acres"][1]
            t[0, i, features.index("archetype_pct_Spatial_Gravity")] = (spatial_grav - norm_dict["archetype_pct_Spatial_Gravity"][0]) / norm_dict["archetype_pct_Spatial_Gravity"][1]
            t[0, i, features.index("period_seq")] = ((i+1) - norm_dict["period_seq"][0]) / norm_dict["period_seq"][1]
            t[0, i, features.index("local_unemployment_rate")] = 0 
            t[0, i, features.index("mortgage_rate_30yr")] = 0
        return torch.tensor(t, dtype=torch.float32)

    pcts = np.arange(0, 105, 5)
    horizons = [5, 10, 15, 20, 25, 30]
    
    fig = go.Figure()
    colorscale = 'Magma' if task_type == "classification" or not "height" in target_col else 'Viridis'
    
    for idx, T in enumerate(horizons):
        periods = np.arange(1, T + 1)
        Z = np.zeros((len(pcts), len(periods)))
        vuln_baseline = create_archetype_tensor(8500.0, 0.9, T)
        
        # Calculate baseline un-shocked height for concession
        if target_col == "proposed_max_height_ft":
            with torch.no_grad():
                baseline_preds = model(vuln_baseline).numpy()[0]
                baseline_unnorm = (baseline_preds[:T] * norm_dict["proposed_max_height_ft"][1]) + norm_dict["proposed_max_height_ft"][0]
                baseline_terminal_height = baseline_unnorm[-1]
        
        for i, pct in enumerate(pcts):
            for j, p in enumerate(periods):
                shock_tensor = vuln_baseline.clone()
                shock_tensor[0, p-1, features.index("petition_pct_this_period")] = float(pct)
                shock_tensor[0, p-1:T, features.index("cumulative_petition_pct")] = float(pct)
                
                with torch.no_grad():
                    preds = model(shock_tensor).numpy()[0]
                    # Also run through survival model
                    if survival_model:
                        surv_logits = survival_model(shock_tensor).numpy()[0]
                        surv_logits_recal = surv_logits[:T] - np.log(K_surv)
                        hazards = 1 / (1 + np.exp(-surv_logits_recal))
                        # Cumulative survival probability vector up to T
                        surv_probs = np.cumprod(1 - hazards)
                    
                if task_type == "classification":
                    logits = preds[:T] - np.log(K_surv)
                    hazards = 1 / (1 + np.exp(-logits))
                    final_val = np.prod(1 - hazards)
                else:
                    if target_col == "proposed_max_height_ft":
                        preds_unnorm = (preds[:T] * norm_dict["proposed_max_height_ft"][1]) + norm_dict["proposed_max_height_ft"][0]
                        shocked_terminal_height = preds_unnorm[-1]
                        # Compute Concession (Feet Lost)
                        final_val = baseline_terminal_height - shocked_terminal_height
                        # Ensure no negative concessions due to noise
                        final_val = max(final_val, 0)
                    else:
                        preds_clip = np.maximum(preds[:T], 0)
                        if is_cumulative:
                            # SURVIVAL ADJUSTMENT: Multiply hearings at time t by probability of surviving to time t
                            adjusted_hearings = preds_clip * surv_probs
                            final_val = np.sum(adjusted_hearings)
                        else:
                            final_val = preds_clip[-1] * surv_probs[-1]
                
                Z[i, j] = final_val
                
        fig.add_trace(go.Surface(
            z=Z, x=periods, y=pcts,
            colorscale=colorscale,
            colorbar=dict(title=z_title),
            visible=(idx == 0),
            name=f"T={T}"
        ))

    print(f"  > Constructing Slider for {target_col}...")
    steps = []
    for i, T in enumerate(horizons):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(horizons)},
                {"title": f'{title}<br><sup>True Interactive Topographical Map Across Timing & Intensity | Evaluated at <b>{T} Periods</b></sup>'}
            ],
            label=f"{T} Periods"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Evaluation Horizon: "},
        pad={"t": 50},
        steps=steps
    )]
    
    z_max = 1.0 if task_type == "classification" else None
    
    fig.update_layout(
        sliders=sliders,
        title=f'{title}<br><sup>True Interactive Topographical Map Across Timing & Intensity | Evaluated at <b>{horizons[0]} Periods</b></sup>',
        scene=dict(
            xaxis=dict(title='Intervention Timing (Period)', range=[1, 30]),
            yaxis=dict(title='Petition Intensity (%)', range=[0, 100]),
            zaxis=dict(title=z_title, range=[0, z_max] if z_max else None),
        ),
        width=1200,
        height=900,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    out_path = rf"{OUT_DIR}\{out_filename}"
    fig.write_html(out_path)
    print(f"  > Master artifact saved to {out_path}")

def main():
    print("1. Loading Dataset for 4D Sweep...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", 
        "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", 
        "period_seq", "petition_pct_this_period", "cumulative_petition_pct", "bw_sin", "bw_cos"
    ]
    
    for f in features: df[f] = df[f].fillna(0)
    df["resolved"] = df["resolved"].fillna(0).astype(int)
    df["council_hearings_this_period"] = df["council_hearings_this_period"].fillna(0).astype(float)
    df["commission_hearings_this_period"] = df["commission_hearings_this_period"].fillna(0).astype(float)
    
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    print("\n2. Training Master Models...")
    model_surv, K_surv = train_model(df, "resolved", features, "classification")
    model_height, _ = train_model(df, "proposed_max_height_ft", features, "regression")
    model_comm, _ = train_model(df, "commission_hearings_this_period", features, "regression")
    model_counc, _ = train_model(df, "council_hearings_this_period", features, "regression")
    
    print("\n3. Generating 4D Sliders...")
    map_4d_surface(model_surv, model_surv, K_surv, df, "resolved", False, "classification",
        'The "Gravity Well" of the Supermajority Law', 'Terminal Survival Probability', 'causal_lstm_4d_survival.html', features, norm_dict)
        
    map_4d_surface(model_height, model_surv, K_surv, df, "proposed_max_height_ft", False, "regression",
        'The Downzoning Surface (Height Concession)', 'Concession (Feet Lost)', 'causal_lstm_4d_height.html', features, norm_dict)
        
    map_4d_surface(model_comm, model_surv, K_surv, df, "commission_hearings_this_period", True, "regression",
        'The Early Friction Surface (Survival-Adjusted)', 'Cumulative Hearings', 'causal_lstm_4d_commission.html', features, norm_dict)
        
    map_4d_surface(model_counc, model_surv, K_surv, df, "council_hearings_this_period", True, "regression",
        'The Late Political Friction Surface (Survival-Adjusted)', 'Cumulative Hearings', 'causal_lstm_4d_council.html', features, norm_dict)

if __name__ == "__main__":
    main()
