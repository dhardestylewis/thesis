import pandas as pd
import numpy as np
import os

def engineer_advanced_petitions():
    panel_path = r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv'
    petitions_path = r'C:\Users\dhl\data\Thesis\thesis\Scratch\Spatial_Engineering\advanced_geometric_petition_intensity.csv'
    
    print("Loading data...")
    panel = pd.read_csv(panel_path, low_memory=False)
    petitions = pd.read_csv(petitions_path)
    # We care about exact petition pct AND the advanced features
    cols_to_keep = ['case_number', 'label_exact_geometric_petition_pct', 
                    'min_signer_dist', 'max_signer_dist', 'median_signer_dist', 
                    'signers_within_200ft', 'signers_outside_200ft', 
                    'unofficial_protest_intensity', 'signer_distance_vector',
                    'protesting_pct_single_family', 'silent_pct_single_family',
                    'protesting_pct_commercial', 'silent_pct_commercial',
                    'protesting_pct_multifamily', 'silent_pct_multifamily',
                    'protesting_mean_parcel_sqft', 'silent_mean_parcel_sqft',
                    'protester_embed_dim1', 'protester_embed_dim2', 'protester_embed_dim3', 'protester_embed_dim4',
                    'temporal_protesting_pct_sf', 'temporal_silent_pct_sf',
                    'temporal_protesting_pct_com', 'temporal_silent_pct_com',
                    'temporal_protesting_pct_mf', 'temporal_silent_pct_mf',
                    'delta_protesting_friction', 'delta_silent_friction']
    
    petitions = petitions[cols_to_keep].drop_duplicates(subset=['case_number'])
    petitions = petitions.rename(columns={'label_exact_geometric_petition_pct': 'true_petition_pct'})
    
    # Identify injection period
    first_comm = panel[panel['commission_hearings_this_period'] > 0].groupby('case_number')['period_seq'].min().reset_index()
    first_comm = first_comm.rename(columns={'period_seq': 'injection_period'})
    
    petitions = petitions.merge(first_comm, on='case_number', how='left')
    petitions['injection_period'] = petitions['injection_period'].fillna(1).astype(int)
    
    # Create injection maps
    petition_map = petitions.set_index(['case_number', 'injection_period'])['true_petition_pct'].to_dict()
    
    # Initialize advanced feature columns
    adv_features = ['min_signer_dist', 'max_signer_dist', 'median_signer_dist', 
                    'signers_within_200ft', 'signers_outside_200ft', 
                    'unofficial_protest_intensity', 'signer_distance_vector',
                    'protesting_pct_single_family', 'silent_pct_single_family',
                    'protesting_pct_commercial', 'silent_pct_commercial',
                    'protesting_pct_multifamily', 'silent_pct_multifamily',
                    'protesting_mean_parcel_sqft', 'silent_mean_parcel_sqft',
                    'protester_embed_dim1', 'protester_embed_dim2', 'protester_embed_dim3', 'protester_embed_dim4',
                    'temporal_protesting_pct_sf', 'temporal_silent_pct_sf',
                    'temporal_protesting_pct_com', 'temporal_silent_pct_com',
                    'temporal_protesting_pct_mf', 'temporal_silent_pct_mf',
                    'delta_protesting_friction', 'delta_silent_friction']
    
    for f in adv_features:
        panel[f] = 0.0 if f != 'signer_distance_vector' else '[]'
        
    # Maps for advanced features
    adv_maps = {f: petitions.set_index(['case_number', 'injection_period'])[f].to_dict() for f in adv_features}
    
    print(f"Preparing to inject {len(petition_map)} true petition values with advanced spatial vectors...")
    
    panel['petition_pct_this_period'] = 0.0
    
    def apply_injection(row):
        key = (row['case_number'], row['period_seq'])
        return petition_map.get(key, 0.0)
        
    panel['petition_pct_this_period'] = panel.apply(apply_injection, axis=1)
    
    for f in adv_features:
        def apply_adv(row):
            key = (row['case_number'], row['period_seq'])
            val = adv_maps[f].get(key, 0.0 if f != 'signer_distance_vector' else '[]')
            return val
        panel[f + "_this_period"] = panel.apply(apply_adv, axis=1)
    
    print("Forward-filling cumulative features...")
    panel['cumulative_petition_pct'] = panel.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    for f in adv_features:
        if f != 'signer_distance_vector':
            panel['cumulative_' + f] = panel.groupby('case_number')[f + "_this_period"].transform(lambda x: x.cumsum().shift(1).fillna(0))
        else:
            # For JSON string vector, just carry forward the last non-empty one
            panel['cumulative_' + f] = panel[f + "_this_period"].replace('[]', np.nan)
            panel['cumulative_' + f] = panel.groupby('case_number')['cumulative_' + f].transform(lambda x: x.ffill().shift(1).fillna('[]'))
    
    # Drop intermediate columns
    for f in adv_features:
        del panel[f]
        del panel[f + "_this_period"]
    
    final_cases = panel['case_number'].nunique()
    final_protested = panel.groupby('case_number').last()['cumulative_petition_pct'] > 0
    print(f"Final legally protested cases: {final_protested.sum()}")
    
    unofficial_protested = panel.groupby('case_number').last()['cumulative_unofficial_protest_intensity'] > 0
    print(f"Final UNOFFICIALLY protested cases (including >200ft): {unofficial_protested.sum()}")
    
    print("Saving repaired biweekly panel...")
    panel.to_csv(panel_path, index=False)
    print(f"Synced to {panel_path}")

if __name__ == "__main__":
    engineer_advanced_petitions()
