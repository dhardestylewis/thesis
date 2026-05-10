import pandas as pd
import numpy as np

def clean_zoning(z_str):
    if pd.isna(z_str):
        return 'Unknown'
    z = str(z_str).upper().strip()
    # Strip conditional overlays (-CO) or neighborhood plans (-NP) for cleaner OHE
    z = z.split('-CO')[0].split('-NP')[0].split('-H')[0]
    return z

def engineer_dynamic_zoning():
    panel_path = r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv'
    zoning_path = r'c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv'
    
    # Load data
    panel = pd.read_csv(panel_path)
    zoning = pd.read_csv(zoning_path)
    
    # Extract only necessary columns from zoning
    z_sub = zoning[['case_number', 'Requested_Zoning', 'Staff_Recommendation', 'Final_Zoning']].drop_duplicates('case_number')
    
    # Clean the base zoning categories for modeling
    for col in ['Requested_Zoning', 'Staff_Recommendation', 'Final_Zoning']:
        z_sub[f'{col}_clean'] = z_sub[col].apply(clean_zoning)
        
    # Merge onto panel
    panel = panel.merge(z_sub, on='case_number', how='left')
    
    # Initialize dynamic columns as Unknown/NaN
    panel['dynamic_requested_zoning'] = 'Unknown'
    panel['dynamic_staff_rec_zoning'] = 'Unknown'
    panel['dynamic_final_zoning'] = 'Unknown'
    
    # Apply dynamic logic grouped by case
    def apply_dynamics(group):
        # 1. Requested Zoning is known from period 1
        req_z = group['Requested_Zoning_clean'].iloc[0]
        group['dynamic_requested_zoning'] = req_z
        
        # 2. Staff Rec is known at first commission hearing
        staff_z = group['Staff_Recommendation_clean'].iloc[0]
        # Find first commission hearing
        comm_idx = group[group['commission_hearings_this_period'] > 0].index
        if len(comm_idx) > 0:
            first_comm = comm_idx[0]
            group.loc[first_comm:, 'dynamic_staff_rec_zoning'] = staff_z
        else:
            # If no commission hearing, fallback to council hearing
            council_idx = group[group['council_hearings_this_period'] > 0].index
            if len(council_idx) > 0:
                first_council = council_idx[0]
                group.loc[first_council:, 'dynamic_staff_rec_zoning'] = staff_z
                
        # 3. Final Zoning is known at resolution
        final_z = group['Final_Zoning_clean'].iloc[0]
        res_idx = group[(group['resolved'] == 1) | (group['vote_event'] == 1)].index
        if len(res_idx) > 0:
            first_res = res_idx[0]
            group.loc[first_res:, 'dynamic_final_zoning'] = final_z
            
        return group

    print("Applying longitudinal masking logic...")
    panel = panel.groupby('case_number', group_keys=False).apply(apply_dynamics)
    
    # One-Hot Encode Top N categories to keep it matrix-ready
    top_n = 10
    top_categories = panel['dynamic_requested_zoning'].value_counts().index[:top_n].tolist()
    if 'Unknown' in top_categories:
        top_categories.remove('Unknown')
        
    print(f"One-hot encoding top categories: {top_categories}")
    for cat in top_categories:
        # Requested
        panel[f'req_zoning_{cat}'] = (panel['dynamic_requested_zoning'] == cat).astype(int)
        # Staff
        panel[f'staff_rec_{cat}'] = (panel['dynamic_staff_rec_zoning'] == cat).astype(int)
        # Final
        panel[f'final_zoning_{cat}'] = (panel['dynamic_final_zoning'] == cat).astype(int)
        
    # Drop the static raw columns merged in
    panel = panel.drop(columns=['Requested_Zoning', 'Staff_Recommendation', 'Final_Zoning', 
                                'Requested_Zoning_clean', 'Staff_Recommendation_clean', 'Final_Zoning_clean'])
    
    # Save back to panel
    print("Saving updated biweekly_panel.csv...")
    panel.to_csv(panel_path, index=False)
    
    # Output proof for walkthrough
    shelley = panel[panel['case_number'] == 'C14-2013-0104'][['period_seq', 'commission_hearings_this_period', 'resolved', 'dynamic_requested_zoning', 'dynamic_staff_rec_zoning', 'dynamic_final_zoning']]
    print("\nProof of Time-Varying Logic (Case C14-2013-0104):")
    print(shelley.to_string())

if __name__ == "__main__":
    engineer_dynamic_zoning()
