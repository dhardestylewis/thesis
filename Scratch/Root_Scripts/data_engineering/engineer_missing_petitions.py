import pandas as pd
import numpy as np
import os

def engineer_missing_petitions():
    panel_path = r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv'
    petitions_path = r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\exact_geometric_petition_intensity.csv'
    
    print("Loading data...")
    panel = pd.read_csv(panel_path, low_memory=False)
    petitions = pd.read_csv(petitions_path)
    
    # We only care about the case number and the exact geometric percentage
    petitions = petitions[['case_number', 'label_exact_geometric_petition_pct']].drop_duplicates(subset=['case_number'])
    petitions = petitions.rename(columns={'label_exact_geometric_petition_pct': 'true_petition_pct'})
    
    initial_cases = panel['case_number'].nunique()
    initial_protested = panel.groupby('case_number').last()['cumulative_petition_pct'] > 0
    print(f"Initial protested cases: {initial_protested.sum()}")
    
    # Identify the exact period to inject the petition
    # We inject at the first period where commission_hearings_this_period > 0
    # If no commission hearing, we inject at period_seq == 1
    first_comm = panel[panel['commission_hearings_this_period'] > 0].groupby('case_number')['period_seq'].min().reset_index()
    first_comm = first_comm.rename(columns={'period_seq': 'injection_period'})
    
    # Merge the injection period back to the petition table
    petitions = petitions.merge(first_comm, on='case_number', how='left')
    petitions['injection_period'] = petitions['injection_period'].fillna(1).astype(int)
    
    # Create a mapping for rapid lookup
    petition_map = petitions.set_index(['case_number', 'injection_period'])['true_petition_pct'].to_dict()
    
    print(f"Preparing to inject {len(petition_map)} true petition values...")
    
    # We will completely overwrite the existing corrupted petition columns 
    # (petition_pct_this_period and cumulative_petition_pct) to ensure fidelity.
    panel['petition_pct_this_period'] = 0.0
    
    def apply_injection(row):
        key = (row['case_number'], row['period_seq'])
        return petition_map.get(key, 0.0)
        
    panel['petition_pct_this_period'] = panel.apply(apply_injection, axis=1)
    
    print("Forward-filling cumulative petition percentage...")
    panel['cumulative_petition_pct'] = panel.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    # Verify
    final_cases = panel['case_number'].nunique()
    final_protested = panel.groupby('case_number').last()['cumulative_petition_pct'] > 0
    print(f"Final protested cases: {final_protested.sum()}")
    
    print("Saving repaired biweekly panel...")
    panel.to_csv(panel_path, index=False)
    
    # Also overwrite the one in the AWS deploy folder for consistency
    aws_path = r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv'
    panel.to_csv(aws_path, index=False)
    print(f"Synced to {aws_path}")

if __name__ == "__main__":
    engineer_missing_petitions()
