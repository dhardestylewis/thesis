import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
import functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 1, batch_first=True)
        self.head_surv = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_dg = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_comm = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_counc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        surv_logits = self.head_surv(lstm_out).squeeze(-1)
        surv_probs = torch.sigmoid(surv_logits)
        
        dg_logits = self.head_dg(lstm_out).squeeze(-1)
        comm = self.head_comm(lstm_out).squeeze(-1) * surv_probs
        counc = self.head_counc(lstm_out).squeeze(-1) * surv_probs
        
        return surv_logits, dg_logits, comm, counc

def precompute_tensors(df, features, max_seq):
    groups = df.groupby("case_number")
    cache = []
    
    for _, group in groups:
        group = group.sort_values("period_seq")
        case_date = pd.to_datetime(group["period_start"].values[0])
        seq = group[features].values
        t_surv = group["resolved"].values
        t_dg_terminal = group["t_downgrade"].values[0]
        t_comm = group["commission_hearings_this_period"].values
        t_counc = group["council_hearings_this_period"].values
        
        is_treated = np.max(seq[:, features.index("petition_pct_this_period")]) > 0.0
        
        if len(seq) > max_seq:
            seq = seq[:max_seq]
            t_surv = t_surv[:max_seq]
            t_comm = t_comm[:max_seq]
            t_counc = t_counc[:max_seq]
        if len(seq) < max_seq:
            pad_len = max_seq - len(seq)
            pad_x = np.zeros((pad_len, len(features)))
            pad_y = np.zeros(pad_len)
            seq = np.vstack([seq, pad_x])
            t_surv = np.concatenate([t_surv, pad_y])
            t_comm = np.concatenate([t_comm, pad_y])
            t_counc = np.concatenate([t_counc, pad_y])
            
        cache.append({
            "date": case_date,
            "X": seq,
            "y_surv": t_surv,
            "y_dg": t_dg_terminal,
            "y_comm": t_comm,
            "y_counc": t_counc,
            "is_treated": is_treated
        })
    return cache

def get_train_tensors_from_cache(cache, cutoff_date):
    X_seq, y_surv, y_dg, y_comm, y_counc = [], [], [], [], []
    treated_idx = []
    
    idx = 0
    for item in cache:
        if item["date"] > cutoff_date:
            continue
        X_seq.append(item["X"])
        y_surv.append(item["y_surv"])
        y_dg.append(item["y_dg"])
        y_comm.append(item["y_comm"])
        y_counc.append(item["y_counc"])
        if item["is_treated"]:
            treated_idx.append(idx)
        idx += 1
        
    K = 15
    def oversample(lst):
        return lst + [lst[i] for i in treated_idx] * K

    X_tensor = torch.tensor(np.array(oversample(X_seq)), dtype=torch.float32)
    ys = torch.tensor(np.array(oversample(y_surv)), dtype=torch.float32)
    ydg = torch.tensor(np.array(oversample(y_dg)), dtype=torch.float32)
    ycm = torch.tensor(np.array(oversample(y_comm)), dtype=torch.float32)
    ycc = torch.tensor(np.array(oversample(y_counc)), dtype=torch.float32)
    
    return TensorDataset(X_tensor, ys, ydg, ycm, ycc), K

def train_era_model(dataset, features, device):
    model = MultiTaskLSTM(len(features)).to(device)
    crit_surv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], device=device), reduction='none')
    crit_dg = nn.BCEWithLogitsLoss()
    crit_reg = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scaler = torch.cuda.amp.GradScaler()
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
    
    model.train()
    for epoch in range(15):
        for bx, b_surv, b_dg, b_comm, b_counc in loader:
            bx = bx.to(device)
            b_surv = b_surv.to(device)
            b_dg = b_dg.to(device)
            b_comm = b_comm.to(device)
            b_counc = b_counc.to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                p_surv, p_dg, p_comm, p_counc = model(bx)
                
                mask = (bx[:, :, features.index("period_seq")] != 0).float()
                
                loss_surv = (crit_surv(p_surv, b_surv) * mask).sum()
                loss_dg = crit_dg(p_dg[:, -1], b_dg) 
                loss_comm = (crit_reg(p_comm, b_comm) * mask).sum()
                loss_counc = (crit_reg(p_counc, b_counc) * mask).sum()
                
                total_loss = (loss_surv + loss_comm + loss_counc) / mask.sum() + loss_dg
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    model.eval()
    return model

def map_era_slider(models, K_surv, target_name, is_cumulative, eras_dt, title, z_title, out_filename, features, norm_dict, device, emp_df):
    print(f"\n--- Generating Era Slider for: {target_name} ---")
    
    T = 15
    def create_archetype_tensor(sqft, spatial_grav):
        t = np.zeros((1, 30, len(features)))
        for i in range(T):
            t[0, i, features.index("land_acres")] = (sqft - norm_dict["land_acres"][0]) / norm_dict["land_acres"][1]
            t[0, i, features.index("archetype_pct_Spatial_Gravity")] = (spatial_grav - norm_dict["archetype_pct_Spatial_Gravity"][0]) / norm_dict["archetype_pct_Spatial_Gravity"][1]
            t[0, i, features.index("period_seq")] = ((i+1) - norm_dict["period_seq"][0]) / norm_dict["period_seq"][1]
            t[0, i, features.index("local_unemployment_rate")] = 0 
            t[0, i, features.index("mortgage_rate_30yr")] = 0
        return torch.tensor(t, dtype=torch.float32)

    pcts = np.arange(0, 105, 5)
    periods = np.arange(1, T + 1)
    vuln_baseline = create_archetype_tensor(8500.0, 0.9)
    
    fig = go.Figure()
    colorscale = 'Magma' if target_name != "downgrade" else 'Viridis'
    
    grid_points = []
    for pct in pcts:
        for p in periods:
            shock_tensor = vuln_baseline.clone()
            shock_tensor[0, p-1, features.index("petition_pct_this_period")] = float(pct)
            shock_tensor[0, p-1:T, features.index("cumulative_petition_pct")] = float(pct)
            grid_points.append(shock_tensor.squeeze(0))
    batch_shock = torch.stack(grid_points).to(device)
    
    Z_all = []
    global_z_max = 0.0
    for era in eras_dt:
        model = models[era.strftime('%Y-%m-%d')].to(device)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_surv, pred_dg, pred_comm, pred_counc = model(batch_shock)
            
        if target_name == "survival":
            logits = pred_surv.cpu().numpy()[:, :T] - np.log(K_surv)
            hazards = 1 / (1 + np.exp(-logits))
            final_vals = np.prod(1 - hazards, axis=1)
        elif target_name == "downgrade":
            logits = pred_dg.cpu().numpy()[:, :T]
            probs = 1 / (1 + np.exp(-logits))
            final_vals = probs[:, -1]
        elif target_name == "commission":
            preds_clip = np.maximum(pred_comm.cpu().numpy()[:, :T], 0)
            final_vals = np.sum(preds_clip, axis=1) if is_cumulative else preds_clip[:, -1]
        elif target_name == "council":
            preds_clip = np.maximum(pred_counc.cpu().numpy()[:, :T], 0)
            final_vals = np.sum(preds_clip, axis=1) if is_cumulative else preds_clip[:, -1]
            
        Z_era = final_vals.reshape((len(pcts), len(periods)))
        Z_all.append(Z_era)
        global_z_max = max(global_z_max, np.max(Z_era))
        
    if target_name in ["survival", "downgrade"]: global_z_max = 1.0
    
    eras_str = [d.strftime('%Y-%m-%d') for d in eras_dt]
    
    for idx, (era_dt, era_str, Z) in enumerate(zip(eras_dt, eras_str, Z_all)):
        is_visible = (idx == len(eras_str) - 1)
        fig.add_trace(go.Surface(
            z=Z, x=periods, y=pcts,
            colorscale=colorscale, cmin=0, cmax=global_z_max,
            colorbar=dict(title=z_title), visible=is_visible, name=f"≤ {era_str}", opacity=0.85
        ))
        
        emp_era = emp_df[emp_df['era_cutoff'] <= era_dt]
        fig.add_trace(go.Scatter3d(
            x=emp_era['timing'], y=emp_era['intensity'], z=emp_era[target_name],
            mode='markers',
            marker=dict(size=4, color='cyan' if target_name in ['survival', 'downgrade'] else 'red', opacity=0.7, line=dict(width=1, color='black')),
            visible=is_visible, name=f"Empirical Data (n={len(emp_era)})", hovertemplate="Timing: %{x}<br>Intensity: %{y}%<br>Outcome: %{z}"
        ))

    steps = []
    for i, era_str in enumerate(eras_str):
        vis = [False] * (2 * len(eras_str))
        vis[i*2] = True
        vis[i*2+1] = True
        step = dict(method="update", args=[{"visible": vis}, {"title": f'{title}<br><sup>Network Trained on Era <b>≤ {era_str}</b> | Monthly Overlays</sup>'}], label=f"≤ {era_str}")
        steps.append(step)

    sliders = [dict(active=len(eras_str) - 1, currentvalue={"prefix": "Training Cutoff: "}, pad={"t": 50}, steps=steps)]
    fig.update_layout(
        sliders=sliders,
        title=f'{title}<br><sup>Network Trained on Era <b>≤ {eras_str[-1]}</b> | Monthly Overlays</sup>',
        scene=dict(xaxis=dict(title='Intervention Timing (Period)', range=[1, T]), yaxis=dict(title='Petition Intensity (%)', range=[0, 100]), zaxis=dict(title=z_title, range=[0, global_z_max])),
        width=1200, height=900, margin=dict(l=65, r=50, b=65, t=90)
    )
    out_path = rf"{OUT_DIR}\{out_filename}"
    fig.write_html(out_path)
    print(f"  > Saved Era artifact to {out_path}")

def main():
    print("1. Loading Dataset for HPC Era Ensembling...")
    df = pd.read_csv(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv", low_memory=False)
    
    master = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", low_memory=False)
    import re
    OVERLAY_STRIP = re.compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
    INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
    def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)
    master["case_number"] = master["case_number"].astype(str).str.strip()
    master["req_int"] = master["Requested_Zoning"].apply(get_int)
    master["fin_int"] = master["Final_Zoning"].apply(get_int)
    master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
    master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)
    df["case_number"] = df["case_number"].astype(str).str.strip()
    df = df.merge(master[["case_number", "t_downgrade"]].drop_duplicates("case_number"), on="case_number", how="left")
    df["t_downgrade"] = df["t_downgrade"].fillna(0)
    
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    features = ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period", "cumulative_petition_pct", "bw_sin", "bw_cos"]
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
        
    print("1.5 Processing Empirical Overlay Points...")
    treated = df[df['cumulative_petition_pct'] > 0]
    records = []
    groups = treated.groupby('case_number')
    for case, group in groups:
        group = group.sort_values('period_seq')
        intervention_idx = group[group['petition_pct_this_period'] > 0].index
        if len(intervention_idx) == 0: continue
        timing = group.loc[intervention_idx[0], 'period_seq']
        if timing > 15: continue
        records.append({
            'case_date': pd.to_datetime(group['period_start'].iloc[-1]),
            'timing': timing,
            'intensity': group['cumulative_petition_pct'].max(),
            'survival': group['resolved'].max(),
            'downgrade': group['t_downgrade'].max(),
            'commission': group['commission_hearings_this_period'].sum(),
            'council': group['council_hearings_this_period'].sum()
        })
    emp_df = pd.DataFrame(records)
    emp_df['era_cutoff'] = emp_df['case_date'].dt.to_period('M').dt.end_time

    print("2. Precomputing Tensor Cache (O(1) lookups)...")
    cache = precompute_tensors(df, features, 30)
    
    unique_dates = sorted(pd.to_datetime(df["period_start"]).unique())
    eras_dt_all = [d for d in unique_dates if "2019-01-01" <= str(d)[:10] <= "2020-12-31"]
    
    # DOWNSAMPLE TO MONTHLY (Taking the last biweekly period of each month)
    df_temp = pd.DataFrame({'dt': eras_dt_all})
    df_temp['month'] = df_temp['dt'].dt.to_period('M')
    eras_dt = df_temp.groupby('month')['dt'].max().tolist()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n3. Executing High-Res Monthly Trace ({len(eras_dt)} Models) on {device.upper()}...")
    
    models = {}
    K_surv = 15
    t0 = time.time()
    for i, era_dt in enumerate(eras_dt):
        era_str = era_dt.strftime('%Y-%m-%d')
        t_model_start = time.time()
        dataset, K = get_train_tensors_from_cache(cache, era_dt)
        K_surv = K
        models[era_str] = train_era_model(dataset, features, device)
        elapsed = time.time() - t_model_start
        print(f"  > [{i+1}/{len(eras_dt)}] Trained Model for Era <= {era_str} in {elapsed:.2f} seconds")
        
    print(f"\n[!] Finished tracing {len(eras_dt)} models in {time.time()-t0:.2f} seconds total.")
        
    print("\n4. Generating Monthly Sliders with Empirical Overlays...")
    map_era_slider(models, K_surv, "survival", False, eras_dt, 'The "Gravity Well" of the Supermajority Law', 'Terminal Survival Probability', 'causal_lstm_monthly_overlay_survival.html', features, norm_dict, device, emp_df)
    map_era_slider(models, K_surv, "downgrade", False, eras_dt, 'The Zoning Downgrade Hazard', 'Probability of Downzoning', 'causal_lstm_monthly_overlay_downgrade.html', features, norm_dict, device, emp_df)
    map_era_slider(models, K_surv, "commission", True, eras_dt, 'The Early Friction Surface', 'Cumulative Hearings', 'causal_lstm_monthly_overlay_commission.html', features, norm_dict, device, emp_df)
    map_era_slider(models, K_surv, "council", True, eras_dt, 'The Late Political Friction Surface', 'Cumulative Hearings', 'causal_lstm_monthly_overlay_council.html', features, norm_dict, device, emp_df)

    print("\n5. Serializing Ensemble Checkpoint to Disk...")
    out_pt = rf"{OUT_DIR}\causal_lstm_biweekly_ensemble.pt"
    state_dicts = {era: model.state_dict() for era, model in models.items()}
    torch.save({
        "models": state_dicts,
        "features": features,
        "norm_dict": norm_dict,
        "K_surv": K_surv
    }, out_pt)
    print(f"  > Successfully saved ensemble weights and metadata to {out_pt}")

if __name__ == "__main__":
    main()
