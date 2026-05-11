"""
read_source_docs.py
Actually read the raw transcript text for:
1. Cases that HAVE petitions — does height concession language exist that the regex misses?
2. The 13 Mechanism-A cases — does the transcript confirm the reduction?
3. Cases with high petition dose — what does the document actually say about height?
"""
import pandas as pd
import numpy as np
import re
import textwrap

# Load sources
print("Loading transcripts and panel...")
comm = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\interim\commission_transcripts.csv', low_memory=False)
panel = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv', low_memory=False)

print(f"Commission transcripts: {len(comm)} rows")
print(f"Comm columns: {list(comm.columns)}")
print()

# Get cases with highest petition dose
pet = pd.to_numeric(panel['petition_pct_this_period'], errors='coerce').fillna(0)
cum = pd.to_numeric(panel['cumulative_petition_pct'], errors='coerce').fillna(0)
top_pet = (panel.groupby('case_number')[['petition_pct_this_period','cumulative_petition_pct']]
           .apply(lambda x: x.apply(pd.to_numeric, errors='coerce').max())
           .sort_values('cumulative_petition_pct', ascending=False)
           .head(20))
print("=== TOP PETITIONED CASES ===")
print(top_pet.to_string())
print()

# The one petitioned case with explicit height reduction
target_cases = ['C14-2019-0054']  # petition=7, red_ht=25, req_ht=35

# Also add top 5 most petitioned cases
top5 = top_pet.index[:5].tolist()
target_cases = list(set(target_cases + top5))

print(f"Cases to investigate: {target_cases}")
print()

CASE_PAT = re.compile(
    r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4}(?:\.[A-Z0-9]+)?)',
    re.IGNORECASE
)

# Height-related patterns (what the regex captures)
PAT_REDUCED_TO = re.compile(r'(?:reduced?|limited|capped)\s+to\s+([0-9]+(?:\.[0-9]+)?)\s*(?:feet|foot|ft)', re.IGNORECASE)
PAT_STAFF_HT   = re.compile(r'staff\s+(?:recommends?|rec\.?)\s+[^\n]{0,120}?([0-9]+(?:\.[0-9]+)?)\s*(?:feet|foot|ft)', re.IGNORECASE)
PAT_COMPAT     = re.compile(r'compatibility\s+standard[^\n]{0,100}?([0-9]+(?:\.[0-9]+)?)\s*(?:feet|foot|ft)', re.IGNORECASE)

# What the regex MISSES — broader height language patterns
PAT_BROAD_HEIGHT = re.compile(
    r'(?:height|story|stories|feet|foot|ft)[^\n]{0,80}?([0-9]+(?:\.[0-9]+)?)\s*(?:feet|foot|ft)',
    re.IGNORECASE
)
PAT_MAX_HEIGHT = re.compile(r'maximum\s+height[^\n]{0,60}?([0-9]+)\s*(?:feet|foot|ft)', re.IGNORECASE)
PAT_HEIGHT_LIMIT = re.compile(r'height\s+(?:limit|restriction|cap|maximum)[^\n]{0,60}?([0-9]+)\s*(?:feet|foot|ft)', re.IGNORECASE)
PAT_CONDITION = re.compile(r'(?:condition|covenant|restrict)[^\n]{0,120}?([0-9]+)\s*(?:feet|foot|ft)', re.IGNORECASE)
PAT_APPROVED_WITH = re.compile(r'approved\s+with\s+(?:the\s+)?(?:following\s+)?conditions?[^\n]{0,200}', re.IGNORECASE)
PAT_NEIGHBORHOOD_PLAN = re.compile(r'neighborhood\s+plan[^\n]{0,200}', re.IGNORECASE)

def print_section(title, char='='):
    print()
    print(char * 60)
    print(title)
    print(char * 60)

for target in target_cases:
    print_section(f"CASE: {target}")

    # Panel data for this case
    case_panel = panel[panel['case_number'] == target]
    if len(case_panel) == 0:
        print("  NOT IN PANEL")
        continue

    max_petition = pd.to_numeric(case_panel['petition_pct_this_period'], errors='coerce').max()
    cum_petition = pd.to_numeric(case_panel['cumulative_petition_pct'], errors='coerce').max()
    req_ht = pd.to_numeric(case_panel['pdf_requested_height_ft'], errors='coerce').max()
    red_ht = pd.to_numeric(case_panel['pdf_reduced_to_ft'], errors='coerce').max()
    nhc = pd.to_numeric(case_panel['net_height_change'], errors='coerce').max()
    resolved = case_panel['resolved'].max()

    print(f"  petition_pct: {max_petition:.4f}  cum_petition: {cum_petition:.4f}")
    print(f"  pdf_requested_height_ft: {req_ht}  pdf_reduced_to_ft: {red_ht}  net_height_change: {nhc}")
    print(f"  resolved: {resolved}")
    print()

    # Find this case in commission transcripts
    target_upper = target.upper()
    hits = []
    for i, row in comm.iterrows():
        text = str(row.get('Raw_Text', ''))
        if target_upper in text.upper():
            hits.append((i, text))

    print(f"  Commission transcript docs mentioning this case: {len(hits)}")

    for doc_idx, (i, text) in enumerate(hits[:3]):  # max 3 docs per case
        print()
        print(f"  --- DOC {doc_idx+1} (row {i}) ---")
        fn = comm.iloc[i].get('Filename', 'unknown') if 'Filename' in comm.columns else 'unknown'
        print(f"  Source file: {fn}")

        # Find all mentions of the case in this doc
        for m in CASE_PAT.finditer(text.upper()):
            if target_upper in m.group(1).upper():
                # Extract 800-char window
                start = max(0, m.start() - 100)
                end   = min(len(text), m.start() + 1200)
                window = text[start:end]

                print(f"\n  [WINDOW around case mention at pos {m.start()}]")
                # Print wrapped
                for line in textwrap.wrap(window.replace('\n', ' '), width=100):
                    print(f"    {line}")

                print()
                print("  REGEX HITS in this window:")

                found_reduced = PAT_REDUCED_TO.findall(window)
                found_staff   = PAT_STAFF_HT.findall(window)
                found_compat  = PAT_COMPAT.findall(window)
                found_max_ht  = PAT_MAX_HEIGHT.findall(window)
                found_ht_lim  = PAT_HEIGHT_LIMIT.findall(window)
                found_cond    = PAT_CONDITION.findall(window)
                found_approv  = PAT_APPROVED_WITH.findall(window)

                print(f"    PAT_REDUCED_TO:      {found_reduced}")
                print(f"    PAT_STAFF_HT:        {found_staff}")
                print(f"    PAT_COMPAT:          {found_compat}")
                print(f"    PAT_MAX_HEIGHT:      {found_max_ht}")
                print(f"    PAT_HEIGHT_LIMIT:    {found_ht_lim}")
                print(f"    PAT_CONDITION:       {found_cond}")
                if found_approv:
                    print(f"    PAT_APPROVED_WITH:   {[x[:80] for x in found_approv]}")

                break  # one window per doc is enough

print_section("SUMMARY: Height language patterns missed by current regex")
# Sample ALL petition > 0 cases, search for broad height language
pet_cases = panel.loc[
    pd.to_numeric(panel['petition_pct_this_period'], errors='coerce') > 0,
    'case_number'
].unique()
print(f"\nTotal cases with any petition activity: {len(pet_cases)}")
print("Scanning commission transcripts for height language in these cases...")

missed_patterns = []
for target in pet_cases:
    target_upper = target.upper()
    for i, row in comm.iterrows():
        text = str(row.get('Raw_Text', ''))
        if target_upper not in text.upper():
            continue
        for m in CASE_PAT.finditer(text.upper()):
            if target_upper not in m.group(1):
                continue
            start = max(0, m.start() - 100)
            end   = min(len(text), m.start() + 1200)
            window = text[start:end]

            # Did current regex catch anything?
            current_catch = (PAT_REDUCED_TO.search(window) or
                             PAT_STAFF_HT.search(window))
            # Does broader pattern fire?
            max_ht_hit = PAT_MAX_HEIGHT.search(window)
            ht_lim_hit = PAT_HEIGHT_LIMIT.search(window)
            cond_hit   = PAT_CONDITION.search(window)

            if not current_catch and (max_ht_hit or ht_lim_hit or cond_hit):
                missed_patterns.append({
                    'case': target,
                    'max_height_match': max_ht_hit.group(0)[:60] if max_ht_hit else None,
                    'height_limit_match': ht_lim_hit.group(0)[:60] if ht_lim_hit else None,
                    'condition_match': cond_hit.group(0)[:60] if cond_hit else None,
                })
            break
        break

missed_df = pd.DataFrame(missed_patterns)
print(f"\nCases with petition + height language the current regex MISSES: {len(missed_df)}")
if len(missed_df) > 0:
    print(missed_df.to_string(index=False))
