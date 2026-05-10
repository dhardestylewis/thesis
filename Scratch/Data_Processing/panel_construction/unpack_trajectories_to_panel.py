"""
unpack_trajectories_to_panel.py
================================
Unpacks the Zoning_Trajectory JSON from model_ready_zoning_data.csv and forward-fills
the continuous dimensional targets (proposed height, FAR, bldg_cov_pct) into the 30-step biweekly panel.
Uses the already-calculated `commission_hearings_this_period` and `council_hearings_this_period` flags 
to trigger the step-function updates, bypassing OCR date parsing errors.
"""

import os
import json
import pandas as pd
import numpy as np
import sys

# Import LDC dictionary from the other script
sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scripts\archive\data_harvesters\03_Data_Engineering_and_Panel_Builds")
try:
    from zoning_delta_calculator import extract_metric
except ImportError:
    print("Could not import zoning_delta_calculator")
    exit()

BASE = r"C:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

ZONING_CSV = os.path.join(BASE, "final", "model_ready_zoning_data.csv")
PANEL_CSV = os.path.join(OUT_DIR, "biweekly_panel.csv")

def main():
    print(f"Loading Panel from {PANEL_CSV}...")
    panel = pd.read_csv(PANEL_CSV)
    
    print(f"Loading Zoning Trajectories from {ZONING_CSV}...")
    zoning = pd.read_csv(ZONING_CSV, low_memory=False, usecols=["case_number", "Zoning_Trajectory"])
    
    # Pre-parse JSON trajectories into a lookup dictionary
    print("Parsing JSON trajectories...")
    traj_dict = {}
    for _, row in zoning.dropna(subset=["Zoning_Trajectory"]).iterrows():
        try:
            events = json.loads(row["Zoning_Trajectory"])
            # Separate events by phase for easy lookup
            comm_events = [e for e in events if e.get("phase") == "Commission"]
            coun_events = [e for e in events if e.get("phase") == "Council"]
            traj_dict[row["case_number"]] = {
                "Commission": comm_events[0] if comm_events else None,
                "Council": coun_events[0] if coun_events else None,
                "All": events
            }
        except Exception as e:
            pass
            
    print("Forward-filling dimensional limits across 30-step timelines based on Hearing Events...")
    
    # Sort panel to ensure chronological processing
    panel = panel.sort_values(["case_number", "period_seq"])
    
    new_hts, new_fars, new_covs = [], [], []
    
    current_case = None
    curr_ht, curr_far, curr_cov = np.nan, np.nan, np.nan
    
    for idx, row in panel.iterrows():
        case = row["case_number"]
        
        # Reset state if new case
        if case != current_case:
            current_case = case
            # Base starting height is the static proposed
            curr_ht = row["proposed_max_height_ft"]
            curr_far = row["proposed_max_far"]
            curr_cov = row["proposed_max_bldg_cov_pct"]
            
            # If trajectory has initial zoning, try to use it as base
            if case in traj_dict and traj_dict[case]["All"]:
                first_event = traj_dict[case]["All"][0]
                initial = first_event.get("requested_zoning") or first_event.get("existing_zoning")
                if initial:
                    h = extract_metric(initial, "max_height_ft")
                    f = extract_metric(initial, "max_far")
                    c = extract_metric(initial, "max_bldg_cov_pct")
                    if pd.notna(h): curr_ht = h
                    if pd.notna(f): curr_far = f
                    if pd.notna(c): curr_cov = c
        
        # Trigger Commission Update
        if row.get("commission_hearings_this_period", 0) >= 1 and case in traj_dict:
            comm_event = traj_dict[case]["Commission"]
            if comm_event:
                zone = comm_event.get("staff_recommendation") or comm_event.get("requested_zoning")
                if zone:
                    h = extract_metric(zone, "max_height_ft")
                    f = extract_metric(zone, "max_far")
                    c = extract_metric(zone, "max_bldg_cov_pct")
                    if pd.notna(h): curr_ht = h
                    if pd.notna(f): curr_far = f
                    if pd.notna(c): curr_cov = c
                    
        # Trigger Council Update
        if (row.get("council_hearings_this_period", 0) >= 1 or row.get("vote_event", 0) == 1) and case in traj_dict:
            coun_event = traj_dict[case]["Council"]
            if coun_event:
                zone = coun_event.get("approved_zoning") or coun_event.get("staff_recommendation")
                if zone:
                    h = extract_metric(zone, "max_height_ft")
                    f = extract_metric(zone, "max_far")
                    c = extract_metric(zone, "max_bldg_cov_pct")
                    if pd.notna(h): curr_ht = h
                    if pd.notna(f): curr_far = f
                    if pd.notna(c): curr_cov = c
                    
        new_hts.append(curr_ht)
        new_fars.append(curr_far)
        new_covs.append(curr_cov)
        
    panel["proposed_max_height_ft"] = new_hts
    panel["proposed_max_far"] = new_fars
    panel["proposed_max_bldg_cov_pct"] = new_covs
    
    # Check dynamics
    cases_with_change = (panel.groupby("case_number")["proposed_max_height_ft"].nunique() > 1).sum()
    print(f"\nDynamism Check: {cases_with_change} cases now have height changes mid-sequence!")
    
    # Save back to disk
    out_cols = [c for c in panel.columns if c != "Zoning_Trajectory"]
    panel[out_cols].to_csv(PANEL_CSV, index=False)
    print(f"Successfully unpacked continuous trajectories and saved to {PANEL_CSV}")

if __name__ == "__main__":
    main()
