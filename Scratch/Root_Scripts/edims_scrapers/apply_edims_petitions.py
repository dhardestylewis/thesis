import pandas as pd
import numpy as np

def apply_edims_petitions():
    panel_path = r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv'
    ocr_path = 'Scratch/ocr_petition_results.csv'
    
    print("Loading datasets...")
    panel = pd.read_csv(panel_path)
    ocr = pd.read_csv(ocr_path)
    
    if len(ocr) == 0:
        print("No petitions found in OCR. Using fallback heuristic.")
        return
        
    print(f"Applying precise EDIMS dates for {len(ocr)} cases...")
    
    # Parse dates from Meeting_URL
    # Format: https://www.austintexas.gov/council/2011/20110217-reg -> 2011-02-17
    def extract_date(url):
        # Extract the 8 digit date part before -reg
        date_str = url.split('/')[-1].split('-')[0]
        return pd.to_datetime(date_str, format='%Y%m%d')
        
    ocr['Petition_Date'] = ocr['Meeting_URL'].apply(extract_date)
    
    # Create a mapping of case -> petition date
    petition_map = ocr.set_index('Case_Number')['Petition_Date'].to_dict()
    
    # Reset existing petition events for protested cases
    # We only touch cases that have a petition (cumulative_petition_events > 0 at end of life)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    cases_with_petitions = panel.groupby('case_number')['cumulative_petition_events'].max()
    protested_cases = cases_with_petitions[cases_with_petitions > 0].index
    
    # Reset columns
    panel.loc[panel['case_number'].isin(protested_cases), 'petition_event'] = 0
    
    # Apply precise dates
    applied_count = 0
    fallback_count = 0
    
    for case in protested_cases:
        case_data = panel[panel['case_number'] == case]
        if case_data.empty:
            continue
            
        if case in petition_map:
            p_date = petition_map[case]
            # Find the period containing this date
            # Ensure p_date is aware or naive matching the panel
            if panel['period_start'].dt.tz is not None:
                p_date = p_date.tz_localize('UTC')
                
            # Find first period where period_start >= p_date
            mask = (panel['case_number'] == case) & (panel['period_start'] >= p_date)
            if mask.any():
                target_idx = panel[mask].index[0]
                panel.at[target_idx, 'petition_event'] = 1
                applied_count += 1
            else:
                # Petition date is after the last period, anchor to last period
                target_idx = case_data.index[-1]
                panel.at[target_idx, 'petition_event'] = 1
                applied_count += 1
        else:
            # Fallback to First Council Hearing Heuristic
            mask_council = (panel['case_number'] == case) & (panel['council_hearings_this_period'] > 0)
            if mask_council.any():
                target_idx = panel[mask_council].index[0]
            else:
                mask_comm = (panel['case_number'] == case) & (panel['commission_hearings_this_period'] > 0)
                if mask_comm.any():
                    target_idx = panel[mask_comm].index[0]
                else:
                    target_idx = case_data.index[0]
                    
            panel.at[target_idx, 'petition_event'] = 1
            fallback_count += 1
            
    # Recalculate cumulative
    panel['cumulative_petition_events'] = panel.groupby('case_number')['petition_event'].cumsum()
    
    panel.to_csv(panel_path, index=False)
    print(f"Done! Successfully anchored {applied_count} cases using precise EDIMS OCR dates.")
    print(f"Fell back to heuristic for {fallback_count} cases.")

if __name__ == '__main__':
    apply_edims_petitions()
