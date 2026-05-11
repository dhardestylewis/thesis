"""
investigate_extraction_gaps.py

Two questions:
1. The 2,203 cases (32%) with no Requested_Zoning — why did extraction fail?
   a) Are they in the commission transcripts at all?
   b) If yes, what does the text around them look like — why did the regex miss?
   c) Are they in council transcripts instead?

2. Height signal: how much approved-height signal would closing the loop give us?
   (init_ht -> req_ht -> approved_ht)
"""
import pandas as pd
import numpy as np
import re

mrd  = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv', low_memory=False)
comm = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\interim\commission_transcripts.csv', low_memory=False)
ct   = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\interim\council_transcripts.csv', low_memory=False)

# ── Cases that FAILED extraction ──────────────────────────────────────────────
missing_zone = mrd[mrd['Requested_Zoning'].fillna('').str.strip() == ''].copy()
print(f"Cases with NO Requested_Zoning: {len(missing_zone)} / {len(mrd)} ({len(missing_zone)/len(mrd):.1%})")
print()

# Status breakdown of missing cases
print("Status breakdown of cases missing zone extraction:")
print(missing_zone['detailed_status'].value_counts().to_string())
print()

# Year breakdown — are they older cases?
if 'calendar_year_folder_created' in missing_zone.columns:
    print("Year breakdown of missing zone cases (top 10):")
    print(missing_zone['calendar_year_folder_created'].value_counts().sort_index().to_string())
    print()

# ── Check commission transcript coverage ──────────────────────────────────────
print("Checking commission transcript coverage for missing-zone cases...")

# Build a set of case numbers mentioned in commission transcripts
CASE_PAT = re.compile(r'C14-\d{4}-\d{4}', re.IGNORECASE)
comm_cases = set()
for text in comm['Raw_Text'].dropna():
    for m in CASE_PAT.finditer(str(text)):
        comm_cases.add(m.group(0).upper())

council_cases = set()
for text in ct['Raw_Text'].dropna() if 'Raw_Text' in ct.columns else []:
    for m in CASE_PAT.finditer(str(text)):
        council_cases.add(m.group(0).upper())

# Normalize missing case numbers
def normalize_case(c):
    return re.sub(r'\.0$', '', str(c).strip().upper())

missing_zone['case_norm'] = missing_zone['case_number'].apply(normalize_case)

in_comm   = missing_zone['case_norm'].isin(comm_cases)
in_council = missing_zone['case_norm'].isin(council_cases)

print(f"Missing-zone cases appearing in commission transcripts: {in_comm.sum()} ({in_comm.mean():.1%})")
print(f"Missing-zone cases appearing in council transcripts:   {in_council.sum()} ({in_council.mean():.1%})")
print(f"Missing-zone cases in NEITHER corpus:                  {(~in_comm & ~in_council).sum()} ({(~in_comm & ~in_council).mean():.1%})")
print()

# ── For cases IN commission but still failed — WHY? ──────────────────────────
in_comm_cases = missing_zone[in_comm]['case_norm'].tolist()
print(f"Cases in commission corpus but regex still missed: {len(in_comm_cases)}")
print("Sampling 5 of them to read the actual window text...")
print()

ZONE_RE = r'(?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|NO|IP|CR)(?:\s*-\s*[0-9A-Z]+){0,4}'
PAT_REQ_TO  = re.compile(r'(?:request|rezoning|rezone).{0,50}?(' + ZONE_RE + r').{0,30}?to\b.{0,30}?(' + ZONE_RE + r')', re.IGNORECASE)
PAT_FROM_TO = re.compile(r'from\s+(' + ZONE_RE + r').{0,30}?to\b.{0,30}?(' + ZONE_RE + r')', re.IGNORECASE)
PAT_ANY_ZONE = re.compile(ZONE_RE, re.IGNORECASE)

for target in in_comm_cases[:5]:
    print(f"--- CASE: {target} ---")
    # Find in commission transcripts
    found_doc = False
    for i, row in comm.iterrows():
        text = str(row.get('Raw_Text', ''))
        if target.upper() in text.upper():
            found_doc = True
            fn = row.get('Filename', 'unknown')
            # Find the window
            pos = text.upper().find(target.upper())
            start = max(0, pos - 50)
            end   = min(len(text), pos + 800)
            window = text[start:end]
            print(f"  File: {fn}")
            # Wrap and print
            import textwrap
            for line in textwrap.wrap(window.replace('\n', ' '), width=100):
                print(f"    {line}")
            print()
            print(f"  PAT_REQ_TO hits:   {PAT_REQ_TO.findall(window)}")
            print(f"  PAT_FROM_TO hits:  {PAT_FROM_TO.findall(window)}")
            print(f"  Any zone codes:    {PAT_ANY_ZONE.findall(window)[:10]}")
            print()
            break
    if not found_doc:
        print(f"  Not found in iteration (case_norm mismatch?)")

# ── Height signal potential if we close the loop ─────────────────────────────
print()
print("=" * 60)
print("HEIGHT SIGNAL: What closing the loop would give us")
print("=" * 60)
print()
print("Current state:")
print(f"  Cases with init_ht populated:    {pd.to_numeric(mrd['Initial_max_height_ft'], errors='coerce').notna().sum()}")
print(f"  Cases with req_ht populated:     {pd.to_numeric(mrd['Requested_max_height_ft'], errors='coerce').notna().sum()}")
print(f"  Cases with approved_ht populated:{pd.to_numeric(mrd['Approved_max_height_ft'], errors='coerce').notna().sum()}")
print()

# If we had Final_Zoning for all 272 Approved cases:
# approved_ht = LDC lookup on Final_Zoning
# The height CONCESSION = req_ht - approved_ht (where approved < requested)
req_ht  = pd.to_numeric(mrd['Requested_max_height_ft'], errors='coerce')
init_ht = pd.to_numeric(mrd['Initial_max_height_ft'], errors='coerce')
approved_s = mrd[mrd['detailed_status'] == 'Approved']
req_appr = pd.to_numeric(approved_s['Requested_max_height_ft'], errors='coerce')
init_appr = pd.to_numeric(approved_s['Initial_max_height_ft'], errors='coerce')

both_appr = approved_s[req_appr.notna() & init_appr.notna()]
req_b = pd.to_numeric(both_appr['Requested_max_height_ft'], errors='coerce')
init_b = pd.to_numeric(both_appr['Initial_max_height_ft'], errors='coerce')
delta_b = req_b - init_b

print(f"Among {len(approved_s)} Approved cases:")
print(f"  Both init + req height available: {len(both_appr)}")
print(f"  req > init (upzone requested):    {(delta_b > 0).sum()}")
print(f"  req == init (same height class):  {(delta_b == 0).sum()}")
print()
print("=> If Final_Zoning were available for all 272 Approved cases,")
print("   we could compute: height_concession = max(req_ht - approved_ht, 0)")
print("   This would be the LDC-class concession — coarse (6 buckets) but real.")
print()
print(f"Note: LDC height buckets = [35, 40, 60, 90, 120, 400] ft")
print("  So if requested CS (60ft) and approved LO (40ft) => concession = 20ft")
print("  If requested CS (60ft) and approved GR (60ft) => concession = 0ft")
print("  This is real policy signal even though coarse.")
