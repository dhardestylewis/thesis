import pandas as pd
df = pd.read_csv('biweekly_panel.csv', nrows=1)
check = [
    'council_hearings_this_period', 'commission_hearings_this_period',
    'nlp_total_tokens', 'cumulative_yea_votes', 'cumulative_nay_votes',
    'net_vote_margin', 'yea_votes_this_period', 'nay_votes_this_period',
    'net_height_change', 'proposed_max_far', 'pdf_requested_zoning'
]
for c in check:
    status = "PRESENT" if c in df.columns else "MISSING"
    print(f"  {c}: {status}")
print()
print("All panel columns:")
for c in sorted(df.columns):
    print(f"  {c}")
