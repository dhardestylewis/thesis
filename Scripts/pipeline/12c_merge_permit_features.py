import os
import csv
import pandas as pd
import numpy as np

BASE = r"c:\Users\dhl\data\Thesis\thesis"
ZONING_CSV = os.path.join(BASE, "Data", "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_causal.csv")
PERMIT_PATH = os.path.join(BASE, "Data", "CoA_Open_Data", "Issued_Building_Permits.csv")
PANEL_PATH = os.path.join(BASE, "Data", "Panel", "biweekly_cradle_to_grave_panel.csv")

def safe_pid(x):
    try: return str(int(float(x))).zfill(10)
    except: return None

def load_permit_features():
    print("Loading raw permits and extracting features...")
    permits = {}
    with open(PERMIT_PATH, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        headers = [h.upper().strip() for h in next(reader)]
        
        tcad_idx = headers.index('TCAD_ID') if 'TCAD_ID' in headers else -1
        app_idx = headers.index('APPLIED_DATE') if 'APPLIED_DATE' in headers else -1
        iss_idx = headers.index('ISSUE_DATE') if 'ISSUE_DATE' in headers else -1
        val_idx = headers.index('TOTAL_JOB_VALUATION') if 'TOTAL_JOB_VALUATION' in headers else -1
        type_idx = headers.index('WORK_TYPE') if 'WORK_TYPE' in headers else -1
        
        for row in reader:
            if len(row) <= max(tcad_idx, app_idx, iss_idx, val_idx, type_idx): continue
            
            tcad = str(row[tcad_idx]).strip().split('.')[0].zfill(10)
            if not tcad or tcad == '0000000000': continue
            
            p_type = str(row[type_idx]).upper()
            if 'NEW' not in p_type and 'COMMERCIAL' not in p_type and 'MULTI' not in p_type: continue
            
            app_d = pd.to_datetime(row[app_idx], errors='coerce') if row[app_idx] else pd.NaT
            iss_d = pd.to_datetime(row[iss_idx], errors='coerce') if row[iss_idx] else pd.NaT
            if pd.notna(app_d) and app_d.tzinfo is not None: app_d = app_d.replace(tzinfo=None)
            if pd.notna(iss_d) and iss_d.tzinfo is not None: iss_d = iss_d.replace(tzinfo=None)
            try:
                val = float(str(row[val_idx]).replace('$', '').replace(',', ''))
            except:
                val = 0.0
                
            if tcad not in permits:
                permits[tcad] = {
                    'applied_date_min': pd.NaT,
                    'issue_date_min': pd.NaT,
                    'total_job_valuation': 0.0,
                    'permit_count': 0
                }
            
            permits[tcad]['permit_count'] += 1
            permits[tcad]['total_job_valuation'] += val
            
            if pd.notna(app_d):
                if pd.isna(permits[tcad]['applied_date_min']) or app_d < permits[tcad]['applied_date_min']:
                    permits[tcad]['applied_date_min'] = app_d
            if pd.notna(iss_d):
                if pd.isna(permits[tcad]['issue_date_min']) or iss_d < permits[tcad]['issue_date_min']:
                    permits[tcad]['issue_date_min'] = iss_d

    return permits

if __name__ == "__main__":
    print("1. Extracting Permit Features")
    permit_data = load_permit_features()

    print("2. Mapping to Zoning Cases")
    z = pd.read_csv(ZONING_CSV, low_memory=False)
    if "parcel_id_10" not in z.columns and "tcad_id" in z.columns:
        z = z.rename(columns={"tcad_id": "parcel_id_10"})
    z["parcel_id_10"] = z["parcel_id_10"].map(safe_pid)

    rows = []
    for _, r in z.iterrows():
        case = r["case_number"]
        tcad = r["parcel_id_10"]
        if pd.isna(tcad) or tcad not in permit_data: continue
        
        p = permit_data[tcad]
        delay = np.nan
        if pd.notna(p['issue_date_min']) and pd.notna(p['applied_date_min']):
            delay = (p['issue_date_min'] - p['applied_date_min']).days
            if delay < 0: delay = 0
            
        rows.append({
            "case_number": case,
            "permit_total_job_valuation": p['total_job_valuation'],
            "permit_total_count": p['permit_count'],
            "permit_review_delay_days": delay
        })

    permit_df = pd.DataFrame(rows).drop_duplicates(subset=["case_number"])

    print("3. Loading 1.15GB Biweekly Panel (this takes RAM)...")
    panel = pd.read_csv(PANEL_PATH, low_memory=False)

    print("4. Merging...")
    panel = panel.merge(permit_df, on="case_number", how="left")

    panel["permit_total_count"] = panel["permit_total_count"].fillna(0)
    panel["permit_total_job_valuation"] = panel["permit_total_job_valuation"].fillna(0)

    print("5. Saving Updated Biweekly Panel...")
    panel.to_csv(PANEL_PATH, index=False)
    print(f"Successfully added 3 new features to {len(panel):,} rows.")
