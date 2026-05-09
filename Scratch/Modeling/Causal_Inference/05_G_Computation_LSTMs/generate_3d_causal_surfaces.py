import os, torch, numpy as np, pandas as pd
import plotly.graph_objects as go
import torch.nn as nn

# Constants & Paths
BASE_DIR = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs"
OUT_DIR = rf"{BASE_DIR}\output"
HTML_OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
VAE_PATH = rf"{OUT_DIR}\causal_vae_weights_v22.pt"
LSTM_PATH = rf"{OUT_DIR}\causal_lstm_weights_v22.pt"
PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from causal_seq2seq_cvae import Seq2SeqCVAE, VariableSelectionNetwork

def generate_3d_plots():
    print("[1/3] Loading data and models...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    # === Match v21 feature engineering exactly ===
    import re as _re
    vote_margins_path = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\engineered_vote_margins.csv"
    if os.path.exists(vote_margins_path):
        votes_df = pd.read_csv(vote_margins_path)
        votes_df['year'] = votes_df['source_file'].str.extract(r'^(\d{4})').astype(int)
        yearly_votes = votes_df.groupby(['case_number', 'year']).agg(
            yea_this_year=('yea_votes', 'sum'), nay_this_year=('nay_votes', 'sum')
        ).reset_index()
        df['period_start_dt'] = pd.to_datetime(df['period_start'])
        df['year'] = df['period_start_dt'].dt.year
        df = df.merge(yearly_votes, on=['case_number', 'year'], how='left')
        df['yea_this_year'] = df['yea_this_year'].fillna(0)
        df['nay_this_year'] = df['nay_this_year'].fillna(0)
        df = df.sort_values(['case_number', 'period_seq'])
        df['cumulative_yea_votes'] = df.groupby('case_number')['yea_this_year'].cumsum()
        df['cumulative_nay_votes'] = df.groupby('case_number')['nay_this_year'].cumsum()
        df['net_vote_margin'] = df['cumulative_yea_votes'] - df['cumulative_nay_votes']
    else:
        df['cumulative_yea_votes'] = 0
        df['cumulative_nay_votes'] = 0
        df['net_vote_margin'] = 0
    
    # Leakage fix: zero tokens where no council hearing
    df['council_nlp_total_tokens'] = df['council_nlp_total_tokens'].where(
        df['council_hearings_this_period'] > 0, other=0
    )
    
    # === V22 LAGGED CUMULATIVE FEATURES ===
    df['cumulative_council_hearings_lag1'] = (
        df.groupby('case_number')['council_hearings_this_period']
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )
    df['cumulative_commission_hearings_lag1'] = (
        df.groupby('case_number')['commission_hearings_this_period']
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )
    df['cumulative_council_nlp_lag1'] = (
        df.groupby('case_number')['council_nlp_total_tokens']
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )
    
    df['vote_friction'] = df['vote_event'] * (1 + df['cumulative_nay_votes'].clip(upper=10))
    df['cumulative_vote_friction'] = df.groupby('case_number')['vote_friction'].cumsum()
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    # === DYNAMIC TARGET ENGINEERING (Time-Varying Mediators) ===
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
        "vote_event", "vote_friction", "yea_this_year", "nay_this_year", "censored",
        "cumulative_council_hearings", "cumulative_commission_hearings",
        "Aggregate_Sentiment",
        "max_opponent_experience"
    ])
    
    features = [c for c in df.columns if c not in exclude]
    
    for f in features + targets:
        if f not in df.columns: df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    
    norm_dict = {}
    for f in features + ["net_height_change"]:
        if f in ["proposed_max_height_ft", "council_nlp_total_tokens", "net_vote_margin", "cumulative_yea_votes", "cumulative_nay_votes", "petition_pct_this_period", "cumulative_petition_pct"]:
            continue
        mean_v, std_v = df[f].mean(), df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    for f in ["proposed_max_height_ft", "council_nlp_total_tokens"]:
        df[f] = np.log1p(df[f].clip(lower=0))
    for f in ["commission_hearings_this_period", "council_hearings_this_period"]:
        df[f] = df[f].clip(lower=0, upper=1)

    # Auto-detect feature count from checkpoint to support old weights
    ckpt = torch.load(rf"{BASE_DIR}\causal_seq2seq_weights.pt", map_location='cpu', weights_only=True)
    ckpt_input_dim = ckpt['vsn.grn.0.weight'].shape[1]
    if 'treatment_residual.weight' in ckpt:
        ckpt_input_dim += 2  # Add back the two bypassed treatment features
        
    if ckpt_input_dim != len(features):
        print(f"  [NOTE] Checkpoint has {ckpt_input_dim} features vs {len(features)} — falling back to old feature set.")
        features = features[:ckpt_input_dim]
    
    treat_idx = [
        features.index("petition_pct_this_period"), 
        features.index("cumulative_petition_pct")
    ]
    
    confounder_idx = []
    if "proposed_max_far" in features:
        confounder_idx.append(features.index("proposed_max_far"))
    if "proposed_max_height_ft" in features:
        confounder_idx.append(features.index("proposed_max_height_ft"))
    if "land_acres" in features:
        confounder_idx.append(features.index("land_acres"))
        
    model = Seq2SeqCVAE(
        len(features), 
        treat_idx=treat_idx, 
        confounder_idx=confounder_idx
    ).to(device)
    model.load_state_dict(ckpt)
    model.eval()


    MAX_SEQ = 55
    pet_idx   = features.index("petition_pct_this_period")
    cum_idx   = features.index("cumulative_petition_pct")
    timing    = 5  # Inject petition at t=5

    # === LDC ZONE COHORTS: Join base zone from source data ===
    import re as _re
    SOURCE_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Source_Data\zoning_cases_prefetched_full.csv"
    src = pd.read_csv(SOURCE_PATH, low_memory=False)[['case_number', 'existing_zoning']]

    def extract_base_zone(z):
        if pd.isna(z): return None
        z = str(z).strip().upper()
        return _re.split(r'[\s\-]', z)[0]

    LDC_FAMILIES = {
        'SF':          ['SF', 'MH'],
        'MF':          ['MF'],
        'Commercial':  ['CS', 'GR', 'LR', 'GO', 'LO', 'CH'],
        'Mixed_Use':   ['DMU', 'CBD', 'PUD', 'CMU', 'TOD'],
    }
    def zone_family(code):
        if not code or not isinstance(code, str): return None
        for fam, prefixes in LDC_FAMILIES.items():
            if any(code.startswith(p) for p in prefixes):
                return fam
        return None

    src['base_zone'] = src['existing_zoning'].apply(extract_base_zone)
    src['zone_family'] = src['base_zone'].apply(zone_family)

    # Merge zone family onto panel (case level)
    case_zones = df[['case_number']].drop_duplicates().merge(
        src[['case_number', 'zone_family']], on='case_number', how='left'
    ).set_index('case_number')['zone_family']

    # 4 LDC-citable cohorts (Austin LDC §25-2)
    cohort_labels = [
        "SF (§25-2-200s)",
        "MF (§25-2-300s)",
        "Commercial (§25-2-700s)",
        "Mixed-Use/Downtown (CBD/DMU/PUD)",
    ]
    cohort_fams = ['SF', 'MF', 'Commercial', 'Mixed_Use']
    cohort_x    = [35.0, 60.0, 75.0, 120.0]  # Canonical height representative in ft

    
    # Petition dose axis: 0% to 100% in 11 steps (full range)
    intensities = np.linspace(0.0, 1.0, 11)
    
    target_cfgs = {
        "survival": {"idx": 0, "title": "Causal Friction: Survival Probability",    "z_label": "Hazard Rate (per period)",  "accum": "mean_hazard"},
        "vote":     {"idx": 1, "title": "Causal Friction: Cumulative Vote Friction", "z_label": "Cumulative Vote Friction",  "accum": "terminal_linear"},
        "height":   {"idx": 2, "title": "LDC Height Friction Surface", "z_label": "Net Height Change (ft)", "accum": "terminal_zscore"},
        "tokens":   {"idx": features.index("cumulative_council_nlp_lag1"), "title": "Administrative Drag (Council NLP Tokens)", "z_label": "Cumulative NLP Tokens", "accum": "terminal_feature"},
        "comm":     {"idx": 4, "title": "Commission Hearings Friction",              "z_label": "Cumulative Hearings",        "accum": "sum_sigmoid"},
        "coun":     {"idx": 5, "title": "Council Hearings Friction",                 "z_label": "Cumulative Hearings",        "accum": "sum_sigmoid"},
    }

    print("[2/3] Running G-Computation (Cohort x Dose)...")
    
    for key, cfg in target_cfgs.items():
        # Z matrix: rows = cohorts, cols = dose levels
        Z_p10 = np.zeros((len(cohort_labels), len(intensities)))
        Z_p50 = np.zeros((len(cohort_labels), len(intensities)))
        Z_p90 = np.zeros((len(cohort_labels), len(intensities)))

        for ci, (label, fam) in enumerate(zip(cohort_labels, cohort_fams)):
            cohort_cases = case_zones[case_zones == fam].index.tolist()
            if len(cohort_cases) == 0:
                print(f"    [{key}] Cohort={label} has 0 cases, skipping.")
                continue
            
            # Build actual observed tensors for this cohort (up to 200 cases for speed)
            sub = df[df['case_number'].isin(cohort_cases[:200])].sort_values(['case_number', 'period_seq'])
            cases_in_sub = sub['case_number'].unique()
            
            # Normalize exactly as training script
            sub_norm = sub[features].copy()
            
            # 1. Apply log transforms FIRST
            for f in ["land_acres", "market_value", "appraised_value"]:
                if f in sub_norm.columns:
                    sub_norm[f] = np.log1p(sub_norm[f].clip(lower=0))
                    
            # 2. Standardize
            for f in norm_dict:
                if f in sub_norm.columns:
                    sub_norm[f] = (sub_norm[f] - norm_dict[f][0]) / (norm_dict[f][1] + 1e-8)
                    
            # 3. Custom handling
            if 'proposed_max_height_ft' in sub_norm.columns:
                sub_norm['proposed_max_height_ft'] = np.log1p(sub['proposed_max_height_ft'].clip(lower=0))
            
            feat_arr = sub_norm.values.astype(np.float32)
            case_sizes = sub.groupby('case_number').size()
            
            # Stack into padded tensor
            n_cases = len(cases_in_sub)
            X_cohort = np.zeros((n_cases, MAX_SEQ, len(features)), dtype=np.float32)
            idx = 0
            for k, c in enumerate(cases_in_sub):
                size = case_sizes[c]
                length = min(size, MAX_SEQ)
                X_cohort[k, :length, :] = feat_arr[idx:idx+length]
                idx += size
            X_t = torch.tensor(X_cohort, dtype=torch.float32)
            
            f_coun = features.index("cumulative_council_hearings_lag1")
            f_comm = features.index("cumulative_commission_hearings_lag1")
            f_tok = features.index("cumulative_council_nlp_lag1")
            f_yea = features.index("cumulative_yea_votes")
            f_nay = features.index("cumulative_nay_votes")
            f_margin = features.index("net_vote_margin")
            
            mean_coun, std_coun = norm_dict["cumulative_council_hearings_lag1"]
            mean_comm, std_comm = norm_dict["cumulative_commission_hearings_lag1"]
            mean_tok, std_tok = norm_dict["cumulative_council_nlp_lag1"]

            for di, intensity in enumerate(intensities):
                # === Counterfactual: inject petition dose at t=timing, forward-fill ===
                X_cf = X_t.clone().to(device)
                X_cf[:, timing-1, pet_idx] = float(intensity)
                X_cf[:, timing-1:, cum_idx] = float(intensity)
                
                with torch.no_grad():
                    X_pre = X_cf[:, :4, :]
                    for t in range(timing-1, 54):
                        preds, _, _, _ = model(X_pre, X_cf)
                        preds_t = preds[:, t, :]
                        pred_tok = torch.expm1(preds_t[:, 3])
                        pred_comm = torch.clamp(preds_t[:, 4], 0, 1)
                        pred_coun = torch.clamp(preds_t[:, 5], 0, 1)
                        pred_yea = torch.relu(preds_t[:, 6])
                        pred_nay = torch.relu(preds_t[:, 7])
                        
                        curr_coun = X_cf[:, t, f_coun] * (std_coun + 1e-8) + mean_coun
                        curr_comm = X_cf[:, t, f_comm] * (std_comm + 1e-8) + mean_comm
                        curr_tok = X_cf[:, t, f_tok] * (std_tok + 1e-8) + mean_tok
                        curr_yea = X_cf[:, t, f_yea]
                        curr_nay = X_cf[:, t, f_nay]
                        
                        next_coun = curr_coun + pred_coun
                        next_comm = curr_comm + pred_comm
                        next_tok = curr_tok + pred_tok
                        next_yea = curr_yea + pred_yea
                        next_nay = curr_nay + pred_nay
                        
                        X_cf[:, t+1, f_coun] = (next_coun - mean_coun) / (std_coun + 1e-8)
                        X_cf[:, t+1, f_comm] = (next_comm - mean_comm) / (std_comm + 1e-8)
                        X_cf[:, t+1, f_tok] = (next_tok - mean_tok) / (std_tok + 1e-8)
                        X_cf[:, t+1, f_yea] = next_yea
                        X_cf[:, t+1, f_nay] = next_nay
                        X_cf[:, t+1, f_margin] = next_yea - next_nay
                        
                    preds, _, _, _ = model(X_pre, X_cf)  # Final inference pass over rolled-out sequence
                
                if cfg["accum"] == "mean_hazard":
                    val = torch.sigmoid(preds[:, :, cfg["idx"]]).mean(dim=1).cpu().numpy()
                elif cfg["accum"] == "terminal_linear":
                    val = preds[:, -1, cfg["idx"]].cpu().numpy()
                elif cfg["accum"] == "terminal_zscore":
                    mean_ht, std_ht = norm_dict["net_height_change"]
                    val = (preds[:, -1, cfg["idx"]].cpu().numpy() * (std_ht + 1e-8)) + mean_ht
                elif cfg["accum"] == "terminal_feature":
                    val_norm = X_cf[:, -1, cfg["idx"]].cpu().numpy()
                    mean_v, std_v = norm_dict["cumulative_council_nlp_lag1"]
                    val = val_norm * (std_v + 1e-8) + mean_v
                elif cfg["accum"] == "sum_sigmoid":
                    val = torch.sigmoid(preds[:, :, cfg["idx"]]).sum(dim=1).cpu().numpy()
                elif cfg["accum"] == "sum_linear":
                    val = preds[:, :, cfg["idx"]].cpu().numpy().sum(axis=1)
                
                Z_p10[ci, di] = np.percentile(val, 10)
                Z_p50[ci, di] = np.percentile(val, 50)
                Z_p90[ci, di] = np.percentile(val, 90)
            
            print(f"    [{key}] Cohort={label:32s} | Dose=0%: P50={Z_p50[ci,0]:.3f}  Dose=50%: P50={Z_p50[ci,5]:.3f}  Dose=100%: P50={Z_p50[ci,10]:.3f}")

        print(f"  > Plotting {cfg['title']}...")
        fig = go.Figure(data=[
            go.Surface(
                # Z shape must be (len_y, len_x) = (n_doses, n_cohorts)
                z=Z_p50.T, x=cohort_x, y=intensities*100,
                colorscale='Viridis',
                colorbar=dict(title=cfg["z_label"]),
                name="P50 (Median)"
            ),
            go.Surface(
                z=Z_p90.T, x=cohort_x, y=intensities*100,
                colorscale='Greys', opacity=0.25, showscale=False,
                name="P90 (Upper Bound)"
            ),
            go.Surface(
                z=Z_p10.T, x=cohort_x, y=intensities*100,
                colorscale='Greys', opacity=0.25, showscale=False,
                name="P10 (Lower Bound)"
            )
        ])
        # Short display labels for axis ticks
        tick_labels = ["SF", "MF", "Commercial", "Mixed-Use/DT"]
        fig.update_layout(
            title=dict(
                text=f"<b>{cfg['title']}</b><br><sup>G-Computation: LDC Base Zone (Austin §25-2) vs Petition Severity</sup>",
                font=dict(size=22)
            ),
            scene=dict(
                xaxis=dict(
                    title='Base Zone (Austin LDC §25-2)',
                    tickvals=cohort_x,
                    ticktext=tick_labels
                ),
                yaxis_title='Counterfactual Petition Severity (%)',
                zaxis_title=cfg["z_label"],
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
            ),
            width=1200, height=900, template="plotly_dark"
        )
        out_path = rf"{HTML_OUT_DIR}\animated_causal_{key}_surface.html"
        fig.write_html(out_path)
        print(f"  > Saved: {out_path}")

    print("\n[3/4] Generating Continuous FAR x Petition Dose Surfaces...")
    
    far_idx = features.index("proposed_max_far")
    far_mean, far_std = norm_dict["proposed_max_far"]
    
    # Selected targets for the continuous surfaces to avoid overwhelming output
    cont_targets = ["survival", "comm", "coun"]
    
    for ci, (label, fam) in enumerate(zip(cohort_labels, cohort_fams)):
        cohort_cases = case_zones[case_zones == fam].index.tolist()
        if len(cohort_cases) == 0: continue
        
        # Calculate P10 to P90 of proposed_max_far for this specific cohort
        cohort_df = df[df['case_number'].isin(cohort_cases)]
        far_vals = cohort_df.groupby('case_number')['proposed_max_far'].first().dropna()
        far_vals = far_vals[far_vals > 0]
        
        if len(far_vals) < 5: continue
        
        p10, p90 = np.percentile(far_vals, 10), np.percentile(far_vals, 90)
        far_sweeps = np.linspace(p10, p90, 11)
        
        # Build actual observed tensors for this cohort
        sub = cohort_df.sort_values(['case_number', 'period_seq'])
        cases_in_sub = sub['case_number'].unique()[:200]  # Cap for speed
        sub = sub[sub['case_number'].isin(cases_in_sub)]
        
        sub_norm = sub[features].copy()
        
        # 1. Apply log transforms FIRST
        for f in ["land_acres", "market_value", "appraised_value"]:
            if f in sub_norm.columns:
                sub_norm[f] = np.log1p(sub_norm[f].clip(lower=0))
                
        # 2. Standardize
        for f in norm_dict:
            if f in sub_norm.columns:
                sub_norm[f] = (sub_norm[f] - norm_dict[f][0]) / (norm_dict[f][1] + 1e-8)
                
        # 3. Custom handling
        if 'proposed_max_height_ft' in sub_norm.columns:
            sub_norm['proposed_max_height_ft'] = np.log1p(sub['proposed_max_height_ft'].clip(lower=0))
        
        feat_arr = sub_norm.values.astype(np.float32)
        case_sizes = sub.groupby('case_number').size()
        
        n_cases = len(cases_in_sub)
        X_cohort = np.zeros((n_cases, MAX_SEQ, len(features)), dtype=np.float32)
        idx = 0
        for k, c in enumerate(cases_in_sub):
            size = case_sizes[c]
            length = min(size, MAX_SEQ)
            X_cohort[k, :length, :] = feat_arr[idx:idx+length]
            idx += size
        X_t = torch.tensor(X_cohort, dtype=torch.float32)
        
        for key in cont_targets:
            cfg = target_cfgs[key]
            # Z matrix: rows = far sweeps, cols = dose levels
            Z_p50 = np.zeros((len(far_sweeps), len(intensities)))
            
            for fi, f_val in enumerate(far_sweeps):
                f_val_norm = (f_val - far_mean) / (far_std + 1e-8)
                for di, intensity in enumerate(intensities):
                    X_cf = X_t.clone().to(device)
                    # Apply counterfactual petition dose
                    X_cf[:, timing-1, pet_idx] = float(intensity)
                    X_cf[:, timing-1:, cum_idx] = float(intensity)
                    # Apply counterfactual FAR
                    X_cf[:, :, far_idx] = float(f_val_norm)
                    
                    with torch.no_grad():
                        X_pre = X_cf[:, :4, :]
                        for t in range(timing-1, 54):
                            preds, _, _, _ = model(X_pre, X_cf)
                            preds_t = preds[:, t, :]
                            pred_tok = torch.expm1(preds_t[:, 3])
                            pred_comm = torch.sigmoid(preds_t[:, 4])
                            pred_coun = torch.sigmoid(preds_t[:, 5])
                            pred_yea = torch.relu(preds_t[:, 6])
                            pred_nay = torch.relu(preds_t[:, 7])
                            
                            curr_coun = X_cf[:, t, f_coun] * (std_coun + 1e-8) + mean_coun
                            curr_comm = X_cf[:, t, f_comm] * (std_comm + 1e-8) + mean_comm
                            curr_tok = X_cf[:, t, f_tok] * (std_tok + 1e-8) + mean_tok
                            curr_yea = X_cf[:, t, f_yea]
                            curr_nay = X_cf[:, t, f_nay]
                            
                            next_coun = curr_coun + pred_coun
                            next_comm = curr_comm + pred_comm
                            next_tok = curr_tok + pred_tok
                            next_yea = curr_yea + pred_yea
                            next_nay = curr_nay + pred_nay
                            
                            X_cf[:, t+1, f_coun] = (next_coun - mean_coun) / (std_coun + 1e-8)
                            X_cf[:, t+1, f_comm] = (next_comm - mean_comm) / (std_comm + 1e-8)
                            X_cf[:, t+1, f_tok] = (next_tok - mean_tok) / (std_tok + 1e-8)
                            X_cf[:, t+1, f_yea] = next_yea
                            X_cf[:, t+1, f_nay] = next_nay
                            X_cf[:, t+1, f_margin] = next_yea - next_nay
                            
                        preds, _, _, _ = model(X_pre, X_cf)
                        
                        if cfg["accum"] == "mean_hazard":
                            val = torch.sigmoid(preds[:, :, cfg["idx"]]).mean(dim=1).cpu().numpy()
                        elif cfg["accum"] == "terminal_linear":
                            val = preds[:, -1, cfg["idx"]].cpu().numpy()
                        elif cfg["accum"] == "terminal_zscore":
                            mean_ht, std_ht = norm_dict["net_height_change"]
                            val = (preds[:, -1, cfg["idx"]].cpu().numpy() * (std_ht + 1e-8)) + mean_ht
                        elif cfg["accum"] == "sum_sigmoid":
                            val = torch.sigmoid(preds[:, :, cfg["idx"]]).sum(dim=1).cpu().numpy()
                        elif cfg["accum"] == "sum_linear":
                            val = preds[:, :, cfg["idx"]].sum(dim=1).cpu().numpy()
                        elif cfg["accum"] == "terminal_feature":
                            val_norm = X_cf[:, -1, cfg["idx"]].cpu().numpy()
                            # Un-normalize using standard scaler
                            mean_v, std_v = norm_dict["cumulative_council_nlp_lag1"]
                            val = val_norm * (std_v + 1e-8) + mean_v
                            
                        Z_p50[fi, di] = np.percentile(val, 50)
            
            # Plot
            fig = go.Figure(data=[
                go.Surface(
                    z=Z_p50.T, x=far_sweeps, y=intensities*100,
                    colorscale='Magma', colorbar=dict(title=cfg["z_label"]), name="P50"
                )
            ])
            safe_fam = fam.replace("/", "_")
            fig.update_layout(
                title=dict(
                    text=f"<b>{fam}: {cfg['title']}</b><br><sup>Continuous FAR vs Petition Severity</sup>",
                    font=dict(size=22)
                ),
                scene=dict(
                    xaxis_title='Proposed Max FAR',
                    yaxis_title='Counterfactual Petition Severity (%)',
                    zaxis_title=cfg["z_label"],
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
                ),
                width=1000, height=800, template="plotly_dark"
            )
            out_path = rf"{HTML_OUT_DIR}\continuous_far_{safe_fam}_{key}.html"
            fig.write_html(out_path)
        print(f"  > Saved continuous FAR surfaces for {fam}")

    print("\n[4/4] Done. All surfaces updated.")
if __name__ == "__main__":
    generate_3d_plots()
