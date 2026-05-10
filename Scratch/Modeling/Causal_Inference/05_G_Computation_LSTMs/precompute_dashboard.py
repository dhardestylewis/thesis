"""
precompute_dashboard.py  –  Per-Case Prediction Store
======================================================
Runs ONCE. Computes predictions for every case × every dose × every timestep
and stores them in SQLite. ~820k rows total (~1462 cases × 11 doses × 51 t).

After this runs, the Streamlit app can filter to ANY combination of height /
lat-lon / date / district with pure SQL WHERE + GROUP BY — no GPU ever again.

Start: .venv_cuda\\Scripts\\python.exe precompute_dashboard.py
"""
import os, sqlite3, time
import pandas as pd
import numpy as np
import torch

import causal_cfm_cvae
causal_cfm_cvae.PANEL_PATH = "biweekly_panel.csv"
from causal_cfm_cvae import load_data, CausalSeq2SeqCFM

DB_PATH = "dashboard_cache/surfaces.db"
os.makedirs("dashboard_cache", exist_ok=True)


def init_db(con):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS case_meta (
            case_number   TEXT PRIMARY KEY,
            requested_ht  REAL,
            latitude      REAL,
            longitude     REAL,
            council_dist  INT,
            filing_year   INT
        );
        CREATE TABLE IF NOT EXISTS case_preds (
            case_number  TEXT    NOT NULL,
            dose         REAL    NOT NULL,
            t            INTEGER NOT NULL,
            surv         REAL,   -- sigmoid -> approval survival prob [0,1]
            ht_delta     REAL,   -- concession percentage [0,1]
            cum_tok      REAL,   -- cumulative total paperwork tokens (log count)
            cum_comm     REAL,   -- cumulative commission hearings (count)
            cum_coun     REAL,   -- cumulative council hearings (count)
            PRIMARY KEY (case_number, dose, t)
        );
        CREATE TABLE IF NOT EXISTS progress (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    con.commit()


def already_done(con, case_number, dose):
    n = con.execute(
        "SELECT COUNT(*) FROM case_preds WHERE case_number=? AND dose=?",
        (case_number, round(dose, 10))
    ).fetchone()[0]
    return n >= 51  # 51 timesteps (4..54)


def main():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")  # allow concurrent reads while writing
    init_db(con)

    print("Loading data...", flush=True)
    # Using the new CFM load_data which returns 10 variables
    X_all, _, _, features, _, norm_dict, treat_idx, cases, cell_assignments, filing_years = load_data()

    # Build case-level metadata for slider filtering
    # df_raw is needed for some metadata fallbacks, so let's load it manually
    df_raw = pd.read_csv("biweekly_panel.csv")
    meta = pd.DataFrame({
        "case_number":  df_raw["case_number"].unique().astype(str),
    }).set_index("case_number")

    # Austin LDC maximum heights by base zone (feet)
    # Source: Austin Land Development Code Chapter 25-2
    ZONE_HT = {
        "SF-1": 35, "SF-2": 35, "SF-3": 35, "SF-4A": 35, "SF-4B": 35, "SF-5": 35, "SF-6": 35,
        "MF-1": 35, "MF-2": 40, "MF-3": 60, "MF-4": 60, "MF-5": 90, "MF-6": 120,
        "LR": 35, "GR": 60, "CS": 60, "CS-1": 60, "CR": 60,
        "LO": 40, "GO": 40, "MO": 60, "NO": 40,
        "CBD": 999, "DMU": 120, "CH": 60, "MI": 60, "LI": 60, "HI": 999,
        "RR": 35, "AG": 35,
    }

    def zone_to_ht(zoning_str):
        """Extract base zone code (strip -NP, -H, -CO suffixes) and look up LDC height."""
        if pd.isna(zoning_str):
            return np.nan
        base = str(zoning_str).split("-")[0].strip().upper()
        # handle two-part codes like SF-6
        parts = str(zoning_str).strip().upper().split("-")
        two_part = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else parts[0]
        return ZONE_HT.get(two_part, ZONE_HT.get(parts[0], np.nan))

    # Build best-available height: explicit PDF height -> story height -> compatibility -> LDC zone lookup
    ht = df_raw.groupby("case_number")["pdf_requested_height_ft"].max()
    for fallback_col in ["pdf_story_height_ft", "pdf_compatibility_height_ft"]:
        if fallback_col in df_raw.columns:
            ht = ht.fillna(df_raw.groupby("case_number")[fallback_col].max())
    if "pdf_story_count" in df_raw.columns:
        ht = ht.fillna(df_raw.groupby("case_number")["pdf_story_count"].max() * 14)
    # Final fallback: LDC max height for the requested zoning type
    if "pdf_requested_zoning" in df_raw.columns:
        zone_ht = df_raw.groupby("case_number")["pdf_requested_zoning"].first().map(zone_to_ht)
        ht = ht.fillna(zone_ht)

    meta["requested_ht"]     = ht  # feet; NaN only if truly unknowable
    meta["height_requested"] = (df_raw.groupby("case_number")["pdf_requested_height_ft"].max().notna()).astype(int)
    meta["latitude"]      = df_raw.groupby("case_number")["latitude"].first()
    meta["longitude"]     = df_raw.groupby("case_number")["longitude"].first()
    meta["council_dist"]  = df_raw.groupby("case_number")["council_district"].first()
    first_dt              = pd.to_datetime(df_raw.groupby("case_number")["period_start"].min())
    meta["filing_year"]   = first_dt.dt.year

    # Filter to valid window
    meta = meta[(meta["filing_year"] >= 2018) & (meta["filing_year"] <= 2024)]
    all_cases = meta.index.tolist()
    print(f"Total cases: {len(all_cases)}", flush=True)

    # Write metadata
    meta.reset_index().rename(columns={"index": "case_number"}).to_sql(
        "case_meta", con, if_exists="replace", index=False
    )
    con.commit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Initialize the new CFM Model
    model = CausalSeq2SeqCFM(
        input_dim=len(features),
        y_dim=5,
        hidden_dim=256,
        latent_dim=64,
        cfm_hidden=512,
        cfm_layers=5,
        n_layers=3,
        treat_idx=treat_idx,
        f_cum_tok=features.index("cumulative_council_nlp_lag1") if "cumulative_council_nlp_lag1" in features else None,
        f_cum_comm=features.index("cumulative_commission_hearings_lag1") if "cumulative_commission_hearings_lag1" in features else None,
        f_cum_coun=features.index("cumulative_council_hearings_lag1") if "cumulative_council_hearings_lag1" in features else None,
    ).to(device)
    
    # Just load fold 0 for the precompute dashboard (since they are basically ensemble)
    fold = 0
    state_dict = torch.load(f"aws_deploy/causal_cfm_weights_fold_{fold}.pt", map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    pet_idx  = features.index("petition_pct_this_period")
    cum_pet  = features.index("cumulative_petition_pct")

    mean_coun, std_coun = norm_dict["cumulative_council_hearings_lag1"]
    mean_comm, std_comm = norm_dict["cumulative_commission_hearings_lag1"]
    mean_tok,  std_tok  = norm_dict["cumulative_nlp_total_tokens_lag1"]
    mean_ht,   std_ht   = norm_dict["height_concession_pct"]

    doses = np.linspace(0.0, 1.0, 11).tolist()
    MC = 10  # Reduced Monte Carlo paths from 25 to 10 for speed
    BATCH = 64
    
    # ── OPTIMIZATION 1: Kernel Fusion (torch.compile) ──
    # [DISABLED ON WINDOWS] JIT compilation crashes ThreadPoolExecutor without MSVC.
    # Running strictly in eager mode.

    # Prepare batches
    batches = []
    for batch_start in range(0, len(all_cases), BATCH):
        batch_cases = all_cases[batch_start: batch_start + BATCH]
        
        # Filter strictly pending cases
        pending_cases = []
        for c in batch_cases:
            needs = [d for d in doses if not already_done(con, c, d)]
            if needs:
                pending_cases.append((c, needs))
        
        if pending_cases:
            batch_ids = [c[0] for c in pending_cases]
            batch_indices = [list(cases).index(c) for c in batch_ids]
            batches.append((batch_ids, batch_indices))

    total_work = len(all_cases) * len(doses)
    done_work = total_work - (sum(len(b[0]) for b in batches) * len(doses))
    t0 = time.time()

    print(f"Starting parallel inference on {len(batches)} batches...", flush=True)

    # ── OPTIMIZATION 2 & 3: Parallelization & Euler Step Reduction ──
    import concurrent.futures

    def process_batch(batch_tuple):
        batch_ids, batch_indices = batch_tuple
        X_batch = X_all[batch_indices].to(device)
        N = len(X_batch)
        X_mc = X_batch.unsqueeze(1).repeat(1, MC, 1, 1).view(N * MC, X_batch.size(1), X_batch.size(2))

        rows = []
        with torch.inference_mode(): # Faster than no_grad
            for d in doses:
                d_r = round(d, 10)
                # OPTIMIZATION 3: Drop n_steps to 10!
                preds = model.sample(X_mc[:, :4, :], X_mc, dose_val=d, n_steps=10)

                p_surv = torch.clamp(preds[:, :, 0], 0, 1).cpu().numpy()
                p_ht = (preds[:, :, 1].cpu().numpy() * std_ht) + mean_ht
                p_tok = np.clip((preds[:, :, 2].cpu().numpy() * std_tok) + mean_tok, 0, None)
                p_comm = np.clip((preds[:, :, 3].cpu().numpy() * std_comm) + mean_comm, 0, None)
                p_coun = np.clip((preds[:, :, 4].cpu().numpy() * std_coun) + mean_coun, 0, None)

                def avg(ts): return ts.reshape(N, MC, 55).mean(axis=1)
                a_surv, a_ht, a_cum_tok, a_cum_comm, a_cum_coun = map(avg, [p_surv, p_ht, p_tok, p_comm, p_coun])

                for i, case_id in enumerate(batch_ids):
                    for t in range(4, 55):
                        rows.append((
                            case_id, d_r, t,
                            float(a_surv[i, t]), float(a_ht[i, t]),
                            float(a_cum_tok[i, t]), float(a_cum_comm[i, t]), float(a_cum_coun[i, t])
                        ))
        return rows, len(batch_ids)

    # We use ThreadPoolExecutor because PyTorch Neural ODE drops the GIL during C++ ATen ops
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, (rows, num_cases) in enumerate(executor.map(process_batch, batches)):
            con.executemany("INSERT OR REPLACE INTO case_preds VALUES (?,?,?,?,?,?,?,?)", rows)
            con.commit()
            
            done_work += num_cases * len(doses)
            elapsed = time.time() - t0
            rate = done_work / max(elapsed, 1)
            eta = (total_work - done_work) / max(rate, 0.001)
            print(f"  Processed Batch {i+1}/{len(batches)} | "
                  f"{done_work}/{total_work} ({done_work/total_work*100:.1f}%) | "
                  f"ETA {eta/60:.1f} min", flush=True)

    print("\n=== All cases complete! ===", flush=True)
    con.execute("INSERT OR REPLACE INTO progress VALUES ('status','done')")
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
