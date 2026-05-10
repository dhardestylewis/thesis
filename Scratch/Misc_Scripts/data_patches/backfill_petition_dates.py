"""
Re-scan all 558 pages of the omnibus petition PDF and extract
case_number + date from each page header, then backfill nulls in
petition_signers_from_pdf.csv.
"""
import re
import shutil
import pdfplumber
import pandas as pd
from dateutil import parser as dateparser

PDF_PATH  = r"c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\raw_sample_petition_C241282.pdf"
CSV_PATH  = r"c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_from_pdf.csv"
BAK_PATH  = r"c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_from_pdf_ORIGINAL.csv"
OUT_PATH  = r"c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_backfilled.csv"

# Backup original before touching anything
shutil.copy2(CSV_PATH, BAK_PATH)
print(f"Backup saved to {BAK_PATH}")

# ------------------------------------------------------------------
# Regex patterns to match dates in various formats seen in the PDF
# ------------------------------------------------------------------
DATE_PATTERNS = [
    # "Date: July 29, 2008"  /  "Date: Feb. 12, 2009"  / "Date: Sept. 23, 2008"
    r'Date[:\s]+([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})',
    # "Date: 11/6/2024"  / "Date: 9/25/2012"
    r'Date[:\s]+(\d{1,2}/\d{1,2}/\d{4})',
    # bare date on its own line: "11/6/2024"
    r'^(\d{1,2}/\d{1,2}/\d{4})$',
]

CASE_PATTERN = re.compile(
    r'(C\d{2}[A-Z\-]*\d{4}-\d{4}(?:\.\w+)?|C\d{3}-\d{4}-\d{4}|C814-\d{4}-\d{4})',
    re.IGNORECASE
)

def extract_header_info(text):
    """Return (case_number, date_str) from a page's text, or (None, None)."""
    case_number = None
    date_str    = None

    # Case number
    m = CASE_PATTERN.search(text)
    if m:
        case_number = m.group(1).upper()

    # Date – try each pattern in order
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            date_str = m.group(1).strip()
            break

    return case_number, date_str

# ------------------------------------------------------------------
# Scan every page of the omnibus PDF
# ------------------------------------------------------------------
print(f"Scanning PDF …")
header_map = {}   # case_number -> parsed date string

with pdfplumber.open(PDF_PATH) as pdf:
    total = len(pdf.pages)
    for i, page in enumerate(pdf.pages):
        if i % 50 == 0:
            print(f"  Page {i+1}/{total}")
        text = page.extract_text() or ""
        case_num, date_str = extract_header_info(text)
        if case_num and date_str and case_num not in header_map:
            header_map[case_num] = date_str

print(f"\nExtracted headers for {len(header_map)} unique cases:")
for k, v in list(header_map.items())[:10]:
    print(f"  {k}  ->  {v}")

# ------------------------------------------------------------------
# Parse all extracted date strings to ISO format
# ------------------------------------------------------------------
parsed_map = {}
for case, ds in header_map.items():
    try:
        parsed_map[case] = dateparser.parse(ds, fuzzy=True).strftime('%Y-%m-%d')
    except Exception:
        pass

print(f"\nSuccessfully parsed dates for {len(parsed_map)} cases.")

# ------------------------------------------------------------------
# Backfill the existing CSV
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"\nOriginal null-date rows: {df['date'].isna().sum()} / {len(df)}")

# Map case_number → recovered date
df['recovered_date'] = df['case_number'].map(parsed_map)

# Fill null dates with recovered dates
mask = df['date'].isna() & df['recovered_date'].notna()
df.loc[mask, 'date'] = df.loc[mask, 'recovered_date']
df.drop(columns=['recovered_date'], inplace=True)

remaining_nulls = df['date'].isna().sum()
print(f"Remaining null-date rows after backfill: {remaining_nulls} / {len(df)}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
