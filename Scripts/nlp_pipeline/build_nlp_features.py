import pandas as pd
import numpy as np
import re
import gc
import os

def main():
    print("Loading indices to map Doc_IDs to Dates...")
    # Load all indices
    p_idx = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\raw\indices\planning_commission_index.csv')
    z_idx = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\raw\indices\zoning_platting_commission_index.csv')
    c_idx = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\raw\indices\council_minutes_index.csv')

    # Commission dates
    comm_dates = pd.concat([p_idx[['Doc_ID', 'Meeting_Date']], z_idx[['Doc_ID', 'Meeting_Date']]])
    comm_dates['Doc_ID'] = comm_dates['Doc_ID'].astype(str)
    # Clean the ' (Cancelled)' and other anomalies from the date string
    comm_dates['clean_date'] = comm_dates['Meeting_Date'].astype(str).str.replace(r' \(Cancelled\)', '', regex=True)
    date_map = dict(zip(comm_dates['Doc_ID'], pd.to_datetime(comm_dates['clean_date'], errors='coerce')))

    # Council dates
    c_idx['Doc_ID'] = c_idx['Doc_ID'].astype(str)
    c_idx['Date'] = c_idx['Meeting_Text'].apply(lambda x: str(x).split('  ')[0].strip() if '  ' in str(x) else str(x))
    c_date_map = dict(zip(c_idx['Doc_ID'], pd.to_datetime(c_idx['Date'], errors='coerce')))
    
    # Combine the maps
    all_dates = {**date_map, **c_date_map}

    print("Loading data...")
    panel_path = r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv'
    cases = pd.read_csv(panel_path, usecols=['case_number']).drop_duplicates()
    known_cases = set(cases['case_number'].dropna().tolist())
    
    transcripts_path = r'C:\Users\dhl\data\Thesis\thesis\Data\interim\commission_transcripts.csv'
    transcripts = pd.read_csv(transcripts_path)
    transcripts['Raw_Text'] = transcripts['Raw_Text'].fillna('').astype(str)
    
    council_path = r'C:\Users\dhl\data\Thesis\thesis\Data\interim\council_transcripts.csv'
    council_df = pd.read_csv(council_path)
    council_df['Raw_Text'] = council_df['Raw_Text'].fillna('').astype(str)
    
    print(f"Loaded {len(known_cases)} known cases in the panel.")
    
    re_oppose = re.compile(r'(?i)\boppos[eio]+')
    re_traffic = re.compile(r'(?i)\btraffic\b')
    re_density = re.compile(r'(?i)\bdensity\b')
    case_regex = re.compile(r'(C\d+J?-\d{4}-\d{4}(?:\.\d+[A-Za-z]*)?|NPA-\d{4}-\d{4}(?:\.\d+[A-Za-z]*)?)')
    
    events = []
    
    def process_df(df, source_type):
        for idx, row in df.iterrows():
            text = row['Raw_Text']
            filename = str(row['Filename'])
            
            # Extract Doc_ID
            doc_id_match = re.search(r'^\d{4}_(\d+)_', filename)
            if not doc_id_match:
                doc_id_match = re.search(r'(\d{6})', filename)
            
            if not doc_id_match:
                continue
                
            doc_id = doc_id_match.group(1)
            event_date = all_dates.get(doc_id)
            
            if pd.isna(event_date):
                continue
                
            doc_tokens = len(text.split())
            if doc_tokens == 0:
                continue
                
            doc_oppose = len(re_oppose.findall(text))
            doc_traffic = len(re_traffic.findall(text))
            doc_density = len(re_density.findall(text))
            
            found_cases = set(case_regex.findall(text))
            filename_cases = set(case_regex.findall(filename))
            all_found = found_cases.union(filename_cases)
            matched_cases = all_found.intersection(known_cases)
            
            for case in matched_cases:
                events.append({
                    'case_number': case,
                    'event_date': event_date,
                    'source': source_type,
                    'tokens': doc_tokens,
                    'oppose': doc_oppose,
                    'traffic': doc_traffic,
                    'density': doc_density
                })
                
            if idx % 1000 == 0:
                print(f"Processed {idx} documents from {source_type}...")

    print("Extracting features from Commission transcripts...")
    process_df(transcripts, 'commission')
    
    print("Extracting features from Council transcripts...")
    process_df(council_df, 'council')
    
    if len(events) == 0:
        print("No events found!")
        return
        
    events_df = pd.DataFrame(events)
    print(f"Extracted {len(events_df)} case-level NLP hit events.")
    
    events_df.to_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\interim\nlp_event_log.csv', index=False)
    
    print("Merging cumulatively into biweekly_panel.csv...")
    panel = pd.read_csv(panel_path)
    panel['period_start_dt'] = pd.to_datetime(panel['period_start'])
    
    # We want to map cumulative sums for each case at each period_start.
    # To do this efficiently:
    # 1. Sort events by event_date
    events_df = events_df.sort_values(by=['case_number', 'event_date'])
    
    # 2. Group events by case_number
    # We will build a list of panel updates
    
    # Prepare new columns (init to 0)
    nlp_cols = ['nlp_document_count', 'nlp_total_tokens', 'nlp_oppose_hits', 'nlp_traffic_hits', 'nlp_density_hits',
                'council_nlp_document_count', 'council_nlp_total_tokens', 'council_nlp_oppose_hits', 'council_nlp_traffic_hits', 'council_nlp_density_hits']
    
    # Drop existing NLP columns
    panel = panel.drop(columns=[c for c in nlp_cols if c in panel.columns])
        
    # We will use iterrows on grouped events to accumulate and update the panel rows efficiently
    # However, merge_asof is perfect for this!
    
    # Separate commission and council events
    comm_events = events_df[events_df['source'] == 'commission'].copy()
    coun_events = events_df[events_df['source'] == 'council'].copy()
    
    def build_cumulative_features(evt_df, prefix):
        if evt_df.empty:
            return pd.DataFrame()
            
        # Group by case and date to sum multiple events on the exact same day
        daily = evt_df.groupby(['case_number', 'event_date']).agg({
            'tokens': 'sum',
            'oppose': 'sum',
            'traffic': 'sum',
            'density': 'sum',
            'source': 'count' # This is the document count
        }).reset_index().rename(columns={'source': 'document_count'})
        
        daily = daily.sort_values(by=['case_number', 'event_date'])
        
        # Calculate cumulative sums within each case
        daily['cum_document_count'] = daily.groupby('case_number')['document_count'].cumsum()
        daily['cum_tokens'] = daily.groupby('case_number')['tokens'].cumsum()
        daily['cum_oppose'] = daily.groupby('case_number')['oppose'].cumsum()
        daily['cum_traffic'] = daily.groupby('case_number')['traffic'].cumsum()
        daily['cum_density'] = daily.groupby('case_number')['density'].cumsum()
        
        # Rename to final column names
        daily = daily.rename(columns={
            'cum_document_count': f'{prefix}_nlp_document_count',
            'cum_tokens': f'{prefix}_nlp_total_tokens',
            'cum_oppose': f'{prefix}_nlp_oppose_hits',
            'cum_traffic': f'{prefix}_nlp_traffic_hits',
            'cum_density': f'{prefix}_nlp_density_hits'
        })
        
        return daily[['case_number', 'event_date', 
                      f'{prefix}_nlp_document_count', f'{prefix}_nlp_total_tokens', 
                      f'{prefix}_nlp_oppose_hits', f'{prefix}_nlp_traffic_hits', f'{prefix}_nlp_density_hits']]

    comm_cumul = build_cumulative_features(comm_events, 'nlp')
    if 'nlp_nlp_document_count' in comm_cumul.columns:
        comm_cumul = comm_cumul.rename(columns={
            'nlp_nlp_document_count': 'nlp_document_count',
            'nlp_nlp_total_tokens': 'nlp_total_tokens',
            'nlp_nlp_oppose_hits': 'nlp_oppose_hits',
            'nlp_nlp_traffic_hits': 'nlp_traffic_hits',
            'nlp_nlp_density_hits': 'nlp_density_hits'
        })
        
    coun_cumul = build_cumulative_features(coun_events, 'council')

    # To apply to panel, we need a forward-fill merge (merge_asof)
    # Both datasets must be sorted by the date key
    panel = panel.sort_values(by='period_start_dt')
    
    if not comm_cumul.empty:
        comm_cumul = comm_cumul.sort_values(by='event_date')
        panel = pd.merge_asof(
            panel, 
            comm_cumul,
            left_on='period_start_dt',
            right_on='event_date',
            by='case_number',
            direction='backward'
        ).drop(columns=['event_date'])
        
    if not coun_cumul.empty:
        coun_cumul = coun_cumul.sort_values(by='event_date')
        panel = pd.merge_asof(
            panel, 
            coun_cumul,
            left_on='period_start_dt',
            right_on='event_date',
            by='case_number',
            direction='backward'
        ).drop(columns=['event_date'])
        
    # Fill any NaNs that weren't merged yet (before the first event) with 0
    panel[nlp_cols] = panel[nlp_cols].fillna(0)
    
    # Restore original sorting if desired (usually by case_number and period_seq)
    panel = panel.sort_values(['case_number', 'period_seq'])
    panel = panel.drop(columns=['period_start_dt'])
    
    panel.to_csv(panel_path, index=False)
    print(f"Successfully eliminated target leakage and merged cumulative features into {panel_path}.")
    print(panel[nlp_cols].describe())

if __name__ == '__main__':
    main()
