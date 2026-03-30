"""
extract_council_votes.py — Extract council vote data for Austin zoning cases
=============================================================================
Queries the City of Austin SODA APIs for council meeting agenda items associated
with zoning cases, then parses vote tallies, reading stages, and dissenting
council members from the agenda item descriptions.

Phases:
  1. Query SODA APIs for all closed zoning cases (expanded beyond multi-parcel)
  2. Parse vote tallies from Description text using regex
  3. Scrape meeting agenda HTML pages for additional vote outcomes
  4. Output final structured vote dataset

SODA Datasets:
  - akgy-tbxy: Historical City Council Agenda Items (2004–2020)
  - wsf2-3rpw: City Council Update Items (2015–2024)
  - sich-49ay: City Council Items (2024–Present)

Author: Daniel Hardesty Lewis
Created: 2026-03-22
"""

import pandas as pd
import requests
import re
import time
import os
import sys
import json
from collections import defaultdict
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_DIR, "Data", "Zoning_Cases")
PROCESSED_DIR = os.path.join(DATA_DIR, "Processed_Data")
SOURCE_DIR = os.path.join(DATA_DIR, "Source_Data")

# Input files
MULTI_PARCEL_CSV = os.path.join(PROCESSED_DIR, "multi_parcel_closed_2018_2025.csv")
ALL_ZONING_CSV = os.path.join(SOURCE_DIR, "zoning_cases_prefetched_full.csv")
EXISTING_MEETING_DATES = os.path.join(PROCESSED_DIR, "rezoning_meeting_dates.csv")

# Output files
MEETING_DATES_FULL = os.path.join(PROCESSED_DIR, "rezoning_meeting_dates_full.csv")
COUNCIL_VOTES_CSV = os.path.join(PROCESSED_DIR, "council_votes_parsed.csv")
VOTE_SUMMARY_CSV = os.path.join(PROCESSED_DIR, "council_vote_summary.csv")

# Rate limiting
SODA_DELAY = 0.15  # seconds between SODA queries
SCRAPE_DELAY = 0.5  # seconds between HTML page scrapes

# ---------------------------------------------------------------------------
# Regex patterns for vote extraction
# ---------------------------------------------------------------------------
# Matches patterns like "Vote: 11-0", "Vote 9-2", "Vote: 10-0,"
VOTE_PATTERN = re.compile(
    r'Vote[:\s]+(\d{1,2})\s*[-–]\s*(\d{1,2})',
    re.IGNORECASE
)

# Matches reading stages: "First reading approved", "second and third readings",
# "Approved on 1st reading", "2nd/3rd readings"
READING_PATTERN = re.compile(
    r'(?:approved\s+(?:on\s+)?)?'
    r'(first|second|third|1st|2nd|3rd)'
    r'(?:\s+(?:and\s+)?(second|third|2nd|3rd))?'
    r'\s+reading',
    re.IGNORECASE
)

# Extract named council members who voted nay or were absent
NAY_PATTERN = re.compile(
    r'Council\s+Member[s]?\s+([\w\s,\-–and]+?)\s*(?:voted\s+nay|[\-–]\s*nay)',
    re.IGNORECASE
)
ABSENT_PATTERN = re.compile(
    r'Council\s+Member[s]?\s+([\w\s,\-–and]+?)\s*(?:was|were)\s+(?:off\s+the\s+dais|absent)',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# SODA API query functions
# ---------------------------------------------------------------------------
def search_soda(dataset_id, where_clause=None, q=None, limit=50):
    """Query an Austin Open Data SODA endpoint."""
    url = f"https://data.austintexas.gov/resource/{dataset_id}.json"
    params = {"$limit": limit}
    if where_clause:
        params["$where"] = where_clause
    if q:
        params["$q"] = q
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            return []
    except Exception as e:
        return []


def query_case_agenda_items(case_number):
    """Query all 3 SODA datasets for agenda items related to a case number."""
    results = []

    # 1. Historical (2004-2020): akgy-tbxy — has `zoning_case_number` field
    esc = case_number.replace("'", "''")
    res1 = search_soda("akgy-tbxy", f"zoning_case_number like '%{esc}%'")
    for row in res1:
        item_url = ""
        link = row.get("link_to_clerks_website")
        if isinstance(link, dict):
            item_url = link.get("url", "")
        elif isinstance(link, str):
            item_url = link
        results.append({
            "CASE_NUMBER": case_number,
            "Meeting_Date": row.get("meeting_date"),
            "Agenda_Item": row.get("agenda_item_number"),
            "Body": row.get("body", "Austin City Council"),
            "Description": row.get("item_description", ""),
            "Source": "akgy-tbxy",
            "Item_URL": item_url,
        })

    # 2. Council Updates (2015-2024): wsf2-3rpw — search `description`
    res2 = search_soda("wsf2-3rpw", f"description like '%{esc}%'")
    for row in res2:
        item_url = ""
        link = row.get("item_url")
        if isinstance(link, dict):
            item_url = link.get("url", "")
        elif isinstance(link, str):
            item_url = link
        results.append({
            "CASE_NUMBER": case_number,
            "Meeting_Date": row.get("agenda_date"),
            "Agenda_Item": row.get("item_number"),
            "Body": "City Council",
            "Description": row.get("description", ""),
            "Source": "wsf2-3rpw",
            "Item_URL": item_url,
        })

    # 3. Current (2024+): sich-49ay — full-text search ($q)
    res3 = search_soda("sich-49ay", q=case_number)
    for row in res3:
        item_url = ""
        link = row.get("attachments")
        if isinstance(link, dict):
            item_url = link.get("url", "")
        elif isinstance(link, str):
            item_url = link
        desc = row.get("posting_language", "") or row.get("tags", "")
        results.append({
            "CASE_NUMBER": case_number,
            "Meeting_Date": row.get("agenda_date"),
            "Agenda_Item": row.get("item_number"),
            "Body": "City Council",
            "Description": desc,
            "Source": "sich-49ay",
            "Item_URL": item_url,
        })

    return results


# ---------------------------------------------------------------------------
# Vote parsing functions
# ---------------------------------------------------------------------------
def parse_votes_from_description(description):
    """Extract all vote tallies from an agenda item description.

    Returns a list of dicts:
        { 'yes': int, 'no': int, 'reading': str, 'nay_members': str, 'absent_members': str }
    """
    if not description:
        return []

    votes = []
    vote_matches = list(VOTE_PATTERN.finditer(description))

    for vm in vote_matches:
        yes_count = int(vm.group(1))
        no_count = int(vm.group(2))

        # Look for reading stage near this vote mention
        # Search backwards from the vote match position for reading context
        context_before = description[:vm.start()]
        reading_stage = ""
        reading_matches = list(READING_PATTERN.finditer(context_before))
        if reading_matches:
            rm = reading_matches[-1]  # take the closest one before the vote
            stages = []
            for g in [rm.group(1), rm.group(2)]:
                if g:
                    g_norm = g.lower()
                    if g_norm in ("first", "1st"):
                        stages.append("1st")
                    elif g_norm in ("second", "2nd"):
                        stages.append("2nd")
                    elif g_norm in ("third", "3rd"):
                        stages.append("3rd")
            reading_stage = "/".join(stages) if stages else ""

        # Look for nay voters and absent members near this vote
        context_after = description[vm.start():]
        nay_members = ""
        absent_members = ""

        nay_match = NAY_PATTERN.search(context_after)
        if nay_match:
            nay_members = clean_member_names(nay_match.group(1))

        absent_match = ABSENT_PATTERN.search(context_after)
        if absent_match:
            absent_members = clean_member_names(absent_match.group(1))

        # If no nay found after vote, try the broader context
        if not nay_members and no_count > 0:
            nay_match = NAY_PATTERN.search(description)
            if nay_match:
                nay_members = clean_member_names(nay_match.group(1))

        votes.append({
            "yes": yes_count,
            "no": no_count,
            "reading": reading_stage,
            "nay_members": nay_members,
            "absent_members": absent_members,
        })

    return votes


def clean_member_names(raw_str):
    """Clean up extracted council member names."""
    # Remove "and" and clean up
    names = raw_str.strip()
    names = re.sub(r'\s+and\s+', ', ', names)
    names = re.sub(r'\s*,\s*', ', ', names)
    names = names.strip(', ')
    return names


def determine_final_vote(votes_list):
    """From a list of parsed votes for a case, determine the final vote.

    The final vote is typically the last reading (3rd, or 2nd/3rd combined).
    If only one vote exists, use that.
    """
    if not votes_list:
        return None

    # Sort by reading stage priority: 3rd > 2nd/3rd > 2nd > 1st
    reading_priority = {"3rd": 4, "2nd/3rd": 3.5, "2nd": 3, "1st/2nd/3rd": 3.5, "1st": 2, "": 1}

    def priority(v):
        return reading_priority.get(v.get("reading", ""), 0)

    sorted_votes = sorted(votes_list, key=priority, reverse=True)
    return sorted_votes[0]


# ---------------------------------------------------------------------------
# HTML Scraping for agenda pages
# ---------------------------------------------------------------------------
def scrape_agenda_page(url):
    """Scrape an Austin City Council agenda page for vote information.

    Returns the text content around the relevant agenda item.
    """
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, 'html.parser')
        # The anchor in the URL points to the specific agenda item
        # Extract the text content of the page
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 1: Expand SODA coverage
# ---------------------------------------------------------------------------
def phase1_query_soda(case_numbers, existing_df=None):
    """Query SODA APIs for all case numbers, merge with existing data."""
    print(f"\n{'='*60}")
    print("Phase 1: Querying SODA APIs for agenda items")
    print(f"{'='*60}")
    print(f"Total cases to query: {len(case_numbers)}")

    # Track which cases already have data
    existing_cases = set()
    if existing_df is not None and len(existing_df) > 0:
        existing_cases = set(existing_df["CASE_NUMBER"].unique())
        print(f"Cases with existing meeting data: {len(existing_cases)}")

    # Only query cases we don't already have
    new_cases = [c for c in case_numbers if c not in existing_cases]
    print(f"New cases to query: {len(new_cases)}")

    all_results = []

    # Include existing data
    if existing_df is not None:
        all_results.extend(existing_df.to_dict("records"))

    # Query new cases
    found = 0
    for i, case in enumerate(new_cases):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(new_cases)}] Querying {case}...")
        results = query_case_agenda_items(case)
        if results:
            found += 1
            all_results.extend(results)
        time.sleep(SODA_DELAY)

    print(f"\nFound agenda items for {found} new cases (+ {len(existing_cases)} existing)")

    out_df = pd.DataFrame(all_results).drop_duplicates(
        subset=["CASE_NUMBER", "Meeting_Date", "Agenda_Item", "Source"],
        keep="first"
    )
    out_df.to_csv(MEETING_DATES_FULL, index=False)
    print(f"Saved {len(out_df)} records to {MEETING_DATES_FULL}")
    return out_df


# ---------------------------------------------------------------------------
# Phase 2: Parse votes from descriptions
# ---------------------------------------------------------------------------
def phase2_parse_votes(meeting_df):
    """Parse vote tallies from all meeting descriptions."""
    print(f"\n{'='*60}")
    print("Phase 2: Parsing votes from agenda descriptions")
    print(f"{'='*60}")

    vote_records = []

    for _, row in meeting_df.iterrows():
        desc = str(row.get("Description", ""))
        votes = parse_votes_from_description(desc)

        for v in votes:
            vote_records.append({
                "CASE_NUMBER": row["CASE_NUMBER"],
                "Meeting_Date": row.get("Meeting_Date"),
                "Agenda_Item": row.get("Agenda_Item"),
                "Body": row.get("Body"),
                "Source": row.get("Source"),
                "Item_URL": row.get("Item_URL"),
                "vote_yes": v["yes"],
                "vote_no": v["no"],
                "reading_stage": v["reading"],
                "nay_members": v["nay_members"],
                "absent_members": v["absent_members"],
                "vote_tally": f"{v['yes']}-{v['no']}",
            })

    vote_df = pd.DataFrame(vote_records)
    if len(vote_df) > 0:
        print(f"Found {len(vote_df)} vote records across {vote_df['CASE_NUMBER'].nunique()} cases")
    else:
        print("No votes found in descriptions.")

    return vote_df


# ---------------------------------------------------------------------------
# Phase 3: Scrape agenda pages for missing votes
# ---------------------------------------------------------------------------
def phase3_scrape_missing_votes(meeting_df, vote_df):
    """Scrape agenda HTML pages for cases that don't have vote tallies yet."""
    print(f"\n{'='*60}")
    print("Phase 3: Scraping agenda pages for additional vote data")
    print(f"{'='*60}")

    # Find cases with meeting dates but no votes extracted
    cases_with_votes = set(vote_df["CASE_NUMBER"].unique()) if len(vote_df) > 0 else set()
    all_cases = set(meeting_df["CASE_NUMBER"].unique())
    cases_without_votes = all_cases - cases_with_votes

    print(f"Cases with votes already parsed: {len(cases_with_votes)}")
    print(f"Cases with meeting dates but no votes: {len(cases_without_votes)}")

    # Get URLs for cases without votes
    urls_to_scrape = meeting_df[
        meeting_df["CASE_NUMBER"].isin(cases_without_votes)
        & meeting_df["Item_URL"].notna()
        & (meeting_df["Item_URL"] != "")
    ][["CASE_NUMBER", "Meeting_Date", "Agenda_Item", "Body", "Source", "Item_URL"]].copy()

    # Deduplicate — keep the latest meeting date per case (most likely to have final vote)
    urls_to_scrape = urls_to_scrape.sort_values("Meeting_Date", ascending=False)
    urls_to_scrape = urls_to_scrape.drop_duplicates(subset=["CASE_NUMBER"], keep="first")

    print(f"Agenda URLs to scrape: {len(urls_to_scrape)}")

    scraped_votes = []
    for i, (_, row) in enumerate(urls_to_scrape.iterrows()):
        url = row["Item_URL"]
        if not url or not isinstance(url, str) or not url.startswith("http"):
            continue

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(urls_to_scrape)}] Scraping {row['CASE_NUMBER']}...")

        page_text = scrape_agenda_page(url)
        if page_text:
            # Search for vote mentions in the page text near the case number
            case_esc = re.escape(row["CASE_NUMBER"])
            # Find the section around this case number
            case_match = re.search(case_esc, page_text, re.IGNORECASE)
            if case_match:
                # Take ~2000 chars around the match
                start = max(0, case_match.start() - 200)
                end = min(len(page_text), case_match.end() + 2000)
                context = page_text[start:end]

                votes = parse_votes_from_description(context)
                for v in votes:
                    scraped_votes.append({
                        "CASE_NUMBER": row["CASE_NUMBER"],
                        "Meeting_Date": row["Meeting_Date"],
                        "Agenda_Item": row["Agenda_Item"],
                        "Body": row["Body"],
                        "Source": "scraped_" + str(row.get("Source", "")),
                        "Item_URL": url,
                        "vote_yes": v["yes"],
                        "vote_no": v["no"],
                        "reading_stage": v["reading"],
                        "nay_members": v["nay_members"],
                        "absent_members": v["absent_members"],
                        "vote_tally": f"{v['yes']}-{v['no']}",
                    })

        time.sleep(SCRAPE_DELAY)

    scraped_df = pd.DataFrame(scraped_votes)
    if len(scraped_df) > 0:
        print(f"Scraped {len(scraped_df)} additional vote records for {scraped_df['CASE_NUMBER'].nunique()} cases")
    else:
        print("No additional votes found from scraping.")

    return scraped_df


# ---------------------------------------------------------------------------
# Phase 4: Build final vote summary
# ---------------------------------------------------------------------------
def phase4_build_summary(vote_df, scraped_df, multi_parcel_df):
    """Build the final vote dataset and summary statistics."""
    print(f"\n{'='*60}")
    print("Phase 4: Building final vote dataset")
    print(f"{'='*60}")

    # Combine parsed and scraped votes
    all_votes = pd.concat([vote_df, scraped_df], ignore_index=True)

    if len(all_votes) == 0:
        print("WARNING: No votes found at all!")
        return pd.DataFrame()

    # Save all individual vote records
    all_votes.to_csv(COUNCIL_VOTES_CSV, index=False)
    print(f"Saved {len(all_votes)} individual vote records to {COUNCIL_VOTES_CSV}")

    # Build per-case vote summary — determine the final vote for each case
    summary_records = []
    for case_number, group in all_votes.groupby("CASE_NUMBER"):
        votes_list = group.to_dict("records")

        # Get the final/latest vote (highest reading stage or latest date)
        final = determine_final_vote(votes_list)

        # Also get all unique readings
        readings = set()
        for v in votes_list:
            r = v.get("reading_stage", "")
            if r:
                readings.add(r)

        summary_records.append({
            "CASE_NUMBER": case_number,
            "final_vote_yes": final["vote_yes"] if final else None,
            "final_vote_no": final["vote_no"] if final else None,
            "final_vote_tally": final["vote_tally"] if final else None,
            "final_reading_stage": final["reading_stage"] if final else None,
            "final_nay_members": final["nay_members"] if final else None,
            "final_absent_members": final["absent_members"] if final else None,
            "all_readings_observed": "; ".join(sorted(readings)),
            "total_vote_records": len(votes_list),
            "unanimous": (final["vote_no"] == 0) if final else None,
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(VOTE_SUMMARY_CSV, index=False)
    print(f"Saved vote summaries for {len(summary_df)} cases to {VOTE_SUMMARY_CSV}")

    # Merge with multi-parcel cases
    mp_cases = set(multi_parcel_df["CASE_NUMBER"].unique())
    cases_with_votes = set(summary_df["CASE_NUMBER"].unique())
    overlap = mp_cases & cases_with_votes

    print(f"\n--- Coverage Report ---")
    print(f"Multi-parcel closed cases (2018-2025): {len(mp_cases)}")
    print(f"Cases with vote data: {len(cases_with_votes)}")
    print(f"Multi-parcel cases with votes: {len(overlap)}")
    print(f"Multi-parcel cases missing votes: {len(mp_cases - cases_with_votes)}")
    print(f"Vote coverage rate: {len(overlap)/len(mp_cases)*100:.1f}%")

    # Vote breakdown
    if len(summary_df) > 0:
        unanimous = summary_df["unanimous"].sum()
        contested = (~summary_df["unanimous"]).sum()
        print(f"\nVote breakdown:")
        print(f"  Unanimous (X-0): {int(unanimous)}")
        print(f"  Contested: {int(contested)}")
        if contested > 0:
            contested_df = summary_df[~summary_df["unanimous"]]
            print(f"  Contested tallies:")
            for _, r in contested_df.iterrows():
                print(f"    {r['CASE_NUMBER']}: {r['final_vote_tally']} "
                      f"(nay: {r['final_nay_members']})")

    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Council Vote Extraction Pipeline")
    print("=" * 60)

    # Load input data
    print("\nLoading input datasets...")
    multi_parcel_df = pd.read_csv(MULTI_PARCEL_CSV)
    mp_cases = sorted(multi_parcel_df["CASE_NUMBER"].dropna().unique())
    print(f"  multi_parcel_closed_2018_2025: {len(mp_cases)} cases")

    all_zoning_df = pd.read_csv(ALL_ZONING_CSV)
    all_cases = sorted(all_zoning_df["case_number"].dropna().unique())
    print(f"  zoning_cases_prefetched_full: {len(all_cases)} cases")

    # Load existing meeting dates if available
    existing_df = None
    if os.path.exists(EXISTING_MEETING_DATES):
        existing_df = pd.read_csv(EXISTING_MEETING_DATES)
        print(f"  Existing meeting dates: {len(existing_df)} records for "
              f"{existing_df['CASE_NUMBER'].nunique()} cases")

    # Focus on multi-parcel cases first (the analysis dataset) plus
    # any cases from the existing meeting dates file
    target_cases = list(set(mp_cases))
    if existing_df is not None:
        target_cases = list(set(target_cases) | set(existing_df["CASE_NUMBER"].unique()))
    target_cases = sorted(target_cases)
    print(f"\nTarget cases for SODA query: {len(target_cases)}")

    # Phase 1: Expand SODA coverage
    meeting_df = phase1_query_soda(target_cases, existing_df)

    # Phase 2: Parse votes from descriptions
    vote_df = phase2_parse_votes(meeting_df)

    # Phase 3: Scrape agenda pages for additional votes
    scraped_df = phase3_scrape_missing_votes(meeting_df, vote_df)

    # Phase 4: Build final vote dataset
    summary_df = phase4_build_summary(vote_df, scraped_df, multi_parcel_df)

    print(f"\nDone! Output files:")
    print(f"  {MEETING_DATES_FULL}")
    print(f"  {COUNCIL_VOTES_CSV}")
    print(f"  {VOTE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
