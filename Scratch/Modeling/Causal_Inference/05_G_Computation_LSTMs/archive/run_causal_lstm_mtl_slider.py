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

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 1, batch_first=True)
        self.head_surv = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_height = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_comm = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_counc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # 1. Survival Prediction
        surv_logits = self.head_surv(lstm_out).squeeze(-1)
        surv_probs = torch.sigmoid(surv_logits)
        
        # 2. Height Prediction (Not heavily gated by survival as plans remain)
        height = self.head_height(lstm_out).squeeze(-1)
        
        # 3. Structurally Gated Friction (Neural Expected Value)
        comm = self.head_comm(lstm_out).squeeze(-1) * surv_probs
        counc = self.head_counc(lstm_out).squeeze(-1) * surv_probs
        
        return surv_logits, height, comm, counc

def train_mtl_model(df, features):
    print("  > Initializing Multi-Task Competing-Risks Tensor...")
    max_seq = 30
    groups = df.groupby("case_number")
    X_seq, y_surv, y_ht, y_comm, y_counc = [], [], [], [], []
    
    for _, group in groups:
        case_year = group["year"].min()
        if case_year >= 2019:
            continue
            
        seq = group.sort_values("period_seq")[features].values
        t_surv = group.sort_values("period_seq")["resolved"].values
        t_ht = group.sort_values("period_seq")["proposed_max_height_ft"].values
        t_comm = group.sort_values("period_seq")["commission_hearings_this_period"].values
        t_counc = group.sort_values("period_seq")["council_hearings_this_period"].values
        
        if len(seq) > max_seq:
            seq = seq[:max_seq]
            t_surv = t_surv[:max_seq]
            t_ht = t_ht[:max_seq]
            t_comm = t_comm[:max_seq]
            t_counc = t_counc[:max_seq]
        if len(seq) < max_seq:
            pad_len = max_seq - len(seq)
            pad_x = np.zeros((pad_len, len(features)))
            pad_y = np.zeros(pad_len)
            seq = np.vstack([seq, pad_x])
            t_surv = np.concatenate([t_surv, pad_y])
            t_ht = np.concatenate([t_ht, pad_y])
            t_comm = np.concatenate([t_comm, pad_y])
            t_counc = np.concatenate([t_counc, pad_y])
            
        X_seq.append(seq)
        y_surv.append(t_surv)
        y_ht.append(t_ht)
        y_comm.append(t_comm)
        y_counc.append(t_counc)
        
    treated_idx = [i for i, seq in enumerate(X_seq) if np.max(seq[:, features.index("petition_pct_this_period")]) > 0.0]
    K = 15
    
    def oversample(lst):
        return lst + [lst[i] for i in treated_idx] * K

    X_tensor = torch.tensor(np.array(oversample(X_seq)), dtype=torch.float32)
    y_surv_tensor = torch.tensor(np.array(oversample(y_surv)), dtype=torch.float32)
    y_ht_tensor = torch.tensor(np.array(oversample(y_ht)), dtype=torch.float32)
    y_comm_tensor = torch.tensor(np.array(oversample(y_comm)), dtype=torch.float32)
    y_counc_tensor = torch.tensor(np.array(oversample(y_counc)), dtype=torch.float32)
    
    print("  > Training Unified MTL Network...")
    model = MultiTaskLSTM(len(features))
    
    crit_surv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]), reduction='none')
    crit_reg = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    dataset = TensorDataset(X_tensor, y_surv_tensor, y_ht_tensor, y_comm_tensor, y_counc_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(15):
        for bx, by_surv, by_ht, by_comm, by_counc in loader:
            optimizer.zero_grad()
            pred_surv, pred_ht, pred_comm, pred_counc = model(bx)
            
            mask = (bx[:, :, features.index("period_seq")] != 0).float()
            
            loss_surv = (crit_surv(pred_surv, by_surv) * mask).sum()
            loss_ht = (crit_reg(pred_ht, by_ht) * mask).sum()
            loss_comm = (crit_reg(pred_comm, by_comm) * mask).sum()
            loss_counc = (crit_reg(pred_counc, by_counc) * mask).sum()
            
            # Equal weighting for multi-task loss
            total_loss = (loss_surv + loss_ht + loss_comm + loss_counc) / mask.sum()
            total_loss.backward()
            optimizer.step()
            
    model.eval()
    return model, K

def map_mtl_slider(model, K_surv, df, target_name, is_cumulative, title, z_title, out_filename, features, norm_dict):
    print(f"\n--- Generating MTL 4D Slider for: {target_name} ---")
    
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
    colorscale = 'Magma' if target_name != "height" else 'Viridis'
    
    for idx, T in enumerate(horizons):
        periods = np.arange(1, T + 1)
        Z = np.zeros((len(pcts), len(periods)))
        vuln_baseline = create_archetype_tensor(8500.0, 0.9, T)
        
        if target_name == "height":
            with torch.no_grad():
                _, b_ht, _, _ = model(vuln_baseline)
                baseline_terminal_height = (b_ht.numpy()[0][:T][-1] * norm_dict["proposed_max_height_ft"][1]) + norm_dict["proposed_max_height_ft"][0]
        
        for i, pct in enumerate(pcts):
            for j, p in enumerate(periods):
                shock_tensor = vuln_baseline.clone()
                shock_tensor[0, p-1, features.index("petition_pct_this_period")] = float(pct)
                shock_tensor[0, p-1:T, features.index("cumulative_petition_pct")] = float(pct)
                
                with torch.no_grad():
                    pred_surv, pred_ht, pred_comm, pred_counc = model(shock_tensor)
                    
                if target_name == "survival":
                    logits = pred_surv.numpy()[0][:T] - np.log(K_surv)
                    hazards = 1 / (1 + np.exp(-logits))
                    final_val = np.prod(1 - hazards)
                elif target_name == "height":
                    shocked_ht = (pred_ht.numpy()[0][:T][-1] * norm_dict["proposed_max_height_ft"][1]) + norm_dict["proposed_max_height_ft"][0]
                    final_val = max(baseline_terminal_height - shocked_ht, 0)
                elif target_name == "commission":
                    preds_clip = np.maximum(pred_comm.numpy()[0][:T], 0)
                    final_val = np.sum(preds_clip) if is_cumulative else preds_clip[-1]
                elif target_name == "council":
                    preds_clip = np.maximum(pred_counc.numpy()[0][:T], 0)
                    final_val = np.sum(preds_clip) if is_cumulative else preds_clip[-1]
                
                Z[i, j] = final_val
                
        fig.add_trace(go.Surface(
            z=Z, x=periods, y=pcts,
            colorscale=colorscale,
            colorbar=dict(title=z_title),
            visible=(idx == 0),
            name=f"T={T}"
        ))

    steps = []
    for i, T in enumerate(horizons):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(horizons)},
                {"title": f'{title}<br><sup>Multi-Task Network | Evaluated at <b>{T} Periods</b></sup>'}
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
    
    z_max = 1.0 if target_name == "survival" else None
    
    fig.update_layout(
        sliders=sliders,
        title=f'{title}<br><sup>Multi-Task Network | Evaluated at <b>{horizons[0]} Periods</b></sup>',
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
    print(f"  > Saved MTL artifact to {out_path}")

def main():
    print("1. Loading Dataset for Multi-Task Training...")
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
        
    print("\n2. Executing Unified MTL Training...")
    model_mtl, K_surv = train_mtl_model(df, features)
    
    print("\n3. Generating MTL 4D Sliders...")
    map_mtl_slider(model_mtl, K_surv, df, "survival", False, 'The "Gravity Well" of the Supermajority Law', 'Terminal Survival Probability', 'causal_lstm_mtl_survival.html', features, norm_dict)
    map_mtl_slider(model_mtl, K_surv, df, "height", False, 'The Downzoning Surface (Height Concession)', 'Concession (Feet Lost)', 'causal_lstm_mtl_height.html', features, norm_dict)
    map_mtl_slider(model_mtl, K_surv, df, "commission", True, 'The Early Friction Surface', 'Cumulative Hearings', 'causal_lstm_mtl_commission.html', features, norm_dict)
    map_mtl_slider(model_mtl, K_surv, df, "council", True, 'The Late Political Friction Surface', 'Cumulative Hearings', 'causal_lstm_mtl_council.html', features, norm_dict)

if __name__ == "__main__":
    main()
