import os
import sys
import pandas as pd

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")
PANEL_PATH = os.path.join(DATA_DIR, "Panel", "biweekly_panel.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "Panel", "biweekly_panel_macro.csv")

def get_fred_data():
    print("Fetching Federal Funds Effective Rate (FEDFUNDS) from public CSV...")
    # FRED provides direct CSV downloads for public series without an API key!
    fedfunds = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS", 
                           parse_dates=['observation_date'], na_values='.')
    fedfunds.rename(columns={'observation_date': 'date', 'FEDFUNDS': 'fed_funds_rate'}, inplace=True)
    
    print("Fetching Producer Price Index: Construction Materials (WPUSI012011)...")
    ppi = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPUSI012011", 
                      parse_dates=['observation_date'], na_values='.')
    ppi.rename(columns={'observation_date': 'date', 'WPUSI012011': 'construction_ppi'}, inplace=True)

    # Merge macro indicators
    df_macro = pd.merge(fedfunds, ppi, on='date', how='outer').sort_values('date')
    
    # Forward fill missing values since these are reported monthly
    df_macro['fed_funds_rate'] = pd.to_numeric(df_macro['fed_funds_rate'], errors='coerce')
    df_macro['construction_ppi'] = pd.to_numeric(df_macro['construction_ppi'], errors='coerce')
    df_macro = df_macro.ffill()
    
    # Filter to our study period
    df_macro = df_macro[(df_macro['date'] >= '2007-01-01') & (df_macro['date'] <= '2024-12-31')]
    
    return df_macro

def hydrate_panel(df_macro):
    if not os.path.exists(PANEL_PATH):
        print(f"ERROR: Could not find panel data at {PANEL_PATH}")
        print("Creating a dummy panel for demonstration...")
        # Create a dummy for testing if it doesn't exist
        dates = pd.date_range(start='2010-01-01', end='2020-01-01', freq='14D')
        df_panel = pd.DataFrame({'date': dates, 'dummy_case': range(len(dates))})
    else:
        print(f"Loading existing panel data from {PANEL_PATH}...")
        df_panel = pd.read_csv(PANEL_PATH, parse_dates=['period_start'])
    
    # We will do an asof merge to align the biweekly panel dates with the monthly macro dates
    print("Merging macroeconomic features into the panel...")
    df_panel = df_panel.sort_values('period_start')
    df_macro = df_macro.sort_values('date')
    
    df_merged = pd.merge_asof(df_panel, df_macro, left_on='period_start', right_on='date', direction='backward')
    if 'date' in df_merged.columns:
        df_merged.drop(columns=['date'], inplace=True)
    
    print(f"Saving hydrated panel to {OUTPUT_PATH}...")
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print("Hydration complete! Macro features added: 'fed_funds_rate', 'construction_ppi'")

if __name__ == "__main__":
    df_macro = get_fred_data()
    hydrate_panel(df_macro)
