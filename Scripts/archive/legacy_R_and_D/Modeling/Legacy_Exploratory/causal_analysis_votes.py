"""
causal_analysis_votes.py
========================
Priority 3: Causal / Econometric Analysis 
Part 1: How did votes get impacted by # of protest letters?

Extracts council votes from meeting descriptions and correlates voting behavior 
with the volume of protest petitions (signers or area %).
Outputs: 
  - Analysis/Output/Econometrics/votes_vs_protest.csv
  - Analysis/Output/Econometrics/fig10_vote_impact.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

import os, re

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Econometrics")
os.makedirs(OUT_DIR, exist_ok=True)

DATES_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "rezoning_meeting_dates.csv")
PET_CSV = os.path.join(DATA, "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv")

def extract_votes(desc):
    if not isinstance(desc, str): return np.nan, np.nan
    # Match patterns like "Vote: 10-1", "Vote 9-2", "Vote:11-0"
    match = re.search(r'Vote:?\s*(\d+)\s*-\s*(\d+)', desc, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan

def analyze_votes():
    print("=== PRIORITY 3: COUNCIL VOTES VS PROTESTS ===")
    
    # 1. Load meeting dates/votes
    df_dates = pd.read_csv(DATES_CSV)
    df_dates['votes'] = df_dates['Description'].apply(extract_votes)
    df_dates['vote_for'] = df_dates['votes'].apply(lambda x: x[0])
    df_dates['vote_against'] = df_dates['votes'].apply(lambda x: x[1])
    
    # 2. Process vote items
    votes_df = df_dates.dropna(subset=['vote_for', 'vote_against']).copy()
    votes_df['total_votes'] = votes_df['vote_for'] + votes_df['vote_against']
    votes_df['pct_against'] = (votes_df['vote_against'] / votes_df['total_votes']) * 100
    
    # Keep the final/latest vote per case if there are multiple readings
    votes_df['Meeting_Date'] = pd.to_datetime(votes_df['Meeting_Date'])
    votes_df = votes_df.sort_values(by=['CASE_NUMBER', 'Meeting_Date']).drop_duplicates(subset=['CASE_NUMBER'], keep='last')
    
    print(f"Extracted valid vote records for {len(votes_df)} zoning cases.")
    
    # 3. Load protests
    pet = pd.read_csv(PET_CSV)
    pet['case_number'] = pet['case_number'].str.strip()
    votes_df['CASE_NUMBER'] = votes_df['CASE_NUMBER'].str.strip()
    
    # 4. Merge
    # We want to compare protested cases vs non-protested cases (or severity of protest)
    merged = pd.merge(votes_df, pet, left_on='CASE_NUMBER', right_on='case_number', how='left')
    
    # Fill cases with no protest
    merged['protested'] = merged['case_number'].notna()
    merged['pct_signer_area'] = merged['signer_pct'].fillna(0)
    merged['num_signers'] = merged['signers'].fillna(0)
    
    avg_against_protest = merged[merged['protested']]['pct_against'].mean()
    avg_against_no_protest = merged[~merged['protested']]['pct_against'].mean()
    
    print(f"\nAverage % Council Votes AGAINST:")
    print(f"  - Protested Cases:     {avg_against_protest:.1f}%")
    print(f"  - Non-Protested Cases: {avg_against_no_protest:.1f}%")
    
    # 5. Correlation
    protested_only = merged[merged['protested']].dropna(subset=['pct_signer_area', 'pct_against'])
    if len(protested_only) > 1:
        corr = np.corrcoef(protested_only['pct_signer_area'], protested_only['pct_against'])[0, 1]
        print(f"\nCorrelation between Area % and Votes Against (Protested Only): r = {corr:.3f}")
        
    merged.to_csv(os.path.join(OUT_DIR, "votes_vs_protest.csv"), index=False)
    
    # 6. Plot Scatter & Fit
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Only plot protested cases
    protested_only = merged[merged['protested']]
    ax.scatter(protested_only['pct_signer_area'], protested_only['pct_against'], 
               alpha=0.6, s=50, c='crimson', edgecolors='black', label='Protested Cases')
               
    # Add non-protested cases as a blob at 0
    non_protest_only = merged[~merged['protested']]
    if len(non_protest_only) > 0:
        # jitter the x lightly for non-protested cases
        jitter = np.random.normal(0, 0.5, len(non_protest_only))
        ax.scatter(jitter, non_protest_only['pct_against'], 
                   alpha=0.3, s=20, c='gray', label='No Protest (Signer Area = 0%)')
                   
    # Trendline for protested cases
    if len(protested_only) > 2:
        z = np.polyfit(protested_only['pct_signer_area'], protested_only['pct_against'], 1)
        p = np.poly1d(z)
        ax.plot(protested_only['pct_signer_area'], p(protested_only['pct_signer_area']), 
                "r--", linewidth=2, label=f"Trend (slope: {z[0]:.2f})")
                
    ax.set_title("Impact of Protest Severity on Council Votes", fontsize=14, fontweight='bold')
    ax.set_xlabel("Severity of Protest (% Valid Signer Area)")
    ax.set_ylabel("Council Defection (% Votes Against Project)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig10_vote_impact.png"), dpi=150)
    plt.close()
    print("Saved fig10_vote_impact.png")
    
if __name__ == "__main__":
    analyze_votes()
