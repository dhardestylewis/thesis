"""
Pass 2: Fix remaining 124 null-date rows.
- Hardcode dates for cases found in PDF but missed by regex.
- Impute from year column for cases not present in the PDF.
"""
import pandas as pd

CSV_PATH = r"c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_backfilled.csv"

df = pd.read_csv(CSV_PATH)
print(f"Null dates before pass 2: {df['date'].isna().sum()}")

# --- Group 1: Known from PDF scan (regex missed due to case number suffix) ---
# C814-82-006.02 appears as "C814-82-006.02(83)" in PDF; date is 11/7/2024
# C14-99-0069.01: date is 11/7/2024
known_dates = {
    'C814-82-006.02': '2024-11-07',
    'C14-99-0069.01': '2024-11-07',
}
for case, date in known_dates.items():
    mask = df['case_number'].str.upper() == case.upper()
    df.loc[mask & df['date'].isna(), 'date'] = date
    print(f"  Filled {(mask & df['date'].isna()).sum()} rows for {case} with {date}")

# --- Group 2: Not in this PDF — impute as Jan 1 of recorded year (conservative) ---
# These 4 cases (C14-2024-0019, C14-2022-0008, C14-2021-0008, C14-2021-0023)
# have year populated but no PDF source for an exact date.
not_in_pdf = ['C14-2024-0019', 'C14-2022-0008', 'C14-2021-0008', 'C14-2021-0023']
for case in not_in_pdf:
    mask = (df['case_number'].str.upper() == case.upper()) & df['date'].isna() & df['year'].notna()
    df.loc[mask, 'date'] = df.loc[mask, 'year'].apply(lambda y: f"{int(y)}-01-01")
    print(f"  Imputed year-start date for {mask.sum()} rows -> {case}")

print(f"\nNull dates after pass 2: {df['date'].isna().sum()}")
df.to_csv(CSV_PATH, index=False)
print(f"Saved to {CSV_PATH}")
