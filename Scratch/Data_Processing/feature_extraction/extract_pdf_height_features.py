"""
extract_pdf_height_features.py
-------------------------------
Extracts height signals from BOTH commission_transcripts.csv (14,999 docs)
and council_transcripts.csv (376 docs) using regex over raw PDF text.

Each extraction is linked to its meeting_date to enable leakage-safe
merge_asof onto the panel:
  - Commission signals: dated via commission_agendas_cases.csv
  - Council signals:   dated via council_agendas_cases.csv

Leakage contract: source_date <= period_start enforced by merge_asof(backward).
Council signals are later in the process (post-commission) and will only
populate panel rows at or after the council hearing period.

Outputs: Data/interim/pdf_height_features.csv
"""
import re
import sys
import pandas as pd
import numpy as np

BASE            = r"c:\Users\dhl\data\Thesis\thesis\Data"
COMM_TRANS      = BASE + r"\interim\commission_transcripts.csv"
COUNC_TRANS     = BASE + r"\interim\council_transcripts.csv"
COMM_AGENDA     = BASE + r"\interim\commission_agendas_cases.csv"
COUNC_AGENDA    = BASE + r"\interim\council_agendas_cases.csv"
OUT_PATH        = BASE + r"\interim\pdf_height_features.csv"

# ------------------------------------------------------------------
# Austin LDC Ch. 25-2 dimensional standards (hardcoded from code)
# ------------------------------------------------------------------
AUSTIN_LDC_TABLE = {
    "RR":   {"max_height_ft": 35,  "max_far": 0.05},
    "LA":   {"max_height_ft": 35,  "max_far": 0.15},
    "DR":   {"max_height_ft": 35,  "max_far": 0.15},
    "SF-1": {"max_height_ft": 35,  "max_far": 0.20},
    "SF-2": {"max_height_ft": 35,  "max_far": 0.35},
    "SF-3": {"max_height_ft": 35,  "max_far": 0.40},
    "SF-4A":{"max_height_ft": 35,  "max_far": 0.45},
    "SF-4B":{"max_height_ft": 35,  "max_far": 0.45},
    "SF-5": {"max_height_ft": 35,  "max_far": 0.50},
    "SF-6": {"max_height_ft": 35,  "max_far": 0.40},
    "MH":   {"max_height_ft": 35,  "max_far": 0.50},
    "MF-1": {"max_height_ft": 40,  "max_far": 0.50},
    "MF-2": {"max_height_ft": 40,  "max_far": 0.60},
    "MF-3": {"max_height_ft": 40,  "max_far": 0.75},
    "MF-4": {"max_height_ft": 60,  "max_far": 1.00},
    "MF-5": {"max_height_ft": 60,  "max_far": 1.00},
    "MF-6": {"max_height_ft": 90,  "max_far": 3.00},
    "NO":   {"max_height_ft": 35,  "max_far": 0.35},
    "LO":   {"max_height_ft": 40,  "max_far": 0.70},
    "GO":   {"max_height_ft": 60,  "max_far": 1.00},
    "CR":   {"max_height_ft": 35,  "max_far": 0.35},
    "LR":   {"max_height_ft": 40,  "max_far": 0.50},
    "GR":   {"max_height_ft": 60,  "max_far": 1.00},
    "CS":   {"max_height_ft": 60,  "max_far": 2.00},
    "CS-1": {"max_height_ft": 60,  "max_far": 2.00},
    "CH":   {"max_height_ft": 120, "max_far": 3.00},
    "IP":   {"max_height_ft": 60,  "max_far": 1.00},
    "LI":   {"max_height_ft": 60,  "max_far": 1.00},
    "MI":   {"max_height_ft": 60,  "max_far": 2.00},
    "HI":   {"max_height_ft": 60,  "max_far": 2.00},
    "CBD":  {"max_height_ft": 400, "max_far": 8.00},
    "DMU":  {"max_height_ft": 120, "max_far": 5.00},
}

HEIGHT_VALID_MIN = 5
HEIGHT_VALID_MAX = 300


def ldc_lookup_metrics(zone_str):
    """Parse base zoning code, strip overlays, look up LDC max height and FAR."""
    if not zone_str:
        return np.nan, np.nan
    zone_str = re.sub(r"\s+", "", zone_str.upper())
    base = re.match(r"^([A-Z]{1,5}(?:-[0-9A-Z]+)?)", zone_str)
    if not base:
        return np.nan, np.nan
    b = base.group(1)
    if b in AUSTIN_LDC_TABLE:
        return float(AUSTIN_LDC_TABLE[b]["max_height_ft"]), float(AUSTIN_LDC_TABLE[b]["max_far"])
    b2 = re.sub(r"[A-Z]$", "", b)
    if b2 in AUSTIN_LDC_TABLE:
        return float(AUSTIN_LDC_TABLE[b2]["max_height_ft"]), float(AUSTIN_LDC_TABLE[b2]["max_far"])
    return np.nan, np.nan


def clamp_height(val):
    """Return val if within valid ft range, else NaN."""
    try:
        v = float(val)
        return v if HEIGHT_VALID_MIN <= v <= HEIGHT_VALID_MAX else np.nan
    except (TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------------
# Compiled regex patterns
# ------------------------------------------------------------------
CASE_PAT = re.compile(
    r"((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4}(?:\.[A-Z0-9]+)?)",
    re.IGNORECASE
)
BASE_ZONE  = r"(?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|NO|IP|CR)"
ZONE_RE    = BASE_ZONE + r"(?:\s*-\s*[0-9A-Z]+){0,4}"

PAT_REQ_TO      = re.compile(r"(?:request|rezoning|rezone).{0,50}?(" + ZONE_RE + r").{0,30}?to\b.{0,30}?(" + ZONE_RE + r")", re.IGNORECASE)
PAT_FROM_TO     = re.compile(r"from\s+(" + ZONE_RE + r").{0,30}?to\b.{0,30}?(" + ZONE_RE + r")", re.IGNORECASE)
PAT_PROP_ZONE   = re.compile(r"(?:proposed|requesting?)\s+(?:zoning\s+)?(?:of\s+)?(" + ZONE_RE + r")", re.IGNORECASE)
PAT_PROPOSED_HT = re.compile(r"PROPOSED\s+HEIGHT[:\s]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PAT_ALLOWED_HT  = re.compile(r"ALLOWED\s+HEIGHT[:\s]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PAT_REDUCED_TO  = re.compile(r"(?:reduced?|limited|capped)\s+to\s+([0-9]+(?:\.[0-9]+)?)\s*(?:feet|foot|ft)", re.IGNORECASE)
PAT_STORIES     = re.compile(r"([0-9]+)\s*-?\s*stor(?:y|ies)", re.IGNORECASE)
PAT_COMPAT      = re.compile(r"compatibility\s+standard[^\n]{0,100}?([0-9]+(?:\.[0-9]+)?)\s*(?:feet|foot|ft)", re.IGNORECASE)
PAT_STAFF_HT    = re.compile(r"staff\s+(?:recommends?|rec\.?)\s+[^\n]{0,120}?([0-9]+(?:\.[0-9]+)?)\s*(?:feet|foot|ft)", re.IGNORECASE)
PAT_STAFF_ZONE  = re.compile(r"(?i)staff\s*rec.*?(?:recommendation of |for |to )?(" + ZONE_RE + r")")
PAT_ALL_ZONES   = re.compile(r"(" + ZONE_RE + r")")
RESIDENTIAL_KW  = re.compile(r"\b(?:residential|single.?family|townhome|duplex)\b", re.IGNORECASE)


def parse_window(text, pos):
    """Extract all height signals from a 1000-char window around a case match."""
    start  = max(0, pos - 100)
    end    = min(len(text), pos + 900)
    window = text[start:end]

    out = {
        "pdf_requested_zoning":       None,
        "pdf_staff_recommended_zoning": None,
        "pdf_approved_zoning":        None,
        "pdf_existing_zoning":        None,
        "pdf_requested_height_ft":    np.nan,
        "pdf_requested_max_far":      np.nan,
        "pdf_proposed_height_ft":     np.nan,
        "pdf_story_count":            np.nan,
        "pdf_story_height_ft":        np.nan,
        "pdf_reduced_to_ft":          np.nan,
        "pdf_compatibility_height_ft":np.nan,
        "pdf_staff_recommends_ht":    np.nan,
    }

    for pat in [PAT_REQ_TO, PAT_FROM_TO]:
        m = pat.search(window)
        if m:
            if pat == PAT_FROM_TO:
                out["pdf_existing_zoning"] = re.sub(r"\s+", "", m.group(1).upper())
                zone = m.group(2)
            else:
                zone = m.group(1)
            ht, far = ldc_lookup_metrics(zone)
            if not np.isnan(ht):
                out["pdf_requested_zoning"]    = re.sub(r"\s+", "", zone.upper())
                out["pdf_requested_height_ft"] = ht
                out["pdf_requested_max_far"]   = far
                break
    if np.isnan(out["pdf_requested_height_ft"]):
        m = PAT_PROP_ZONE.search(window)
        if m:
            zone = m.group(1)
            ht, far = ldc_lookup_metrics(zone)
            if not np.isnan(ht):
                out["pdf_requested_zoning"]    = re.sub(r"\s+", "", zone.upper())
                out["pdf_requested_height_ft"] = ht
                out["pdf_requested_max_far"]   = far

    # 2. Explicit numeric height from site data table
    for pat in [PAT_PROPOSED_HT, PAT_ALLOWED_HT]:
        m = pat.search(window)
        if m:
            val = clamp_height(m.group(1))
            if not np.isnan(val):
                out["pdf_proposed_height_ft"] = val
                break

    # 3. Story count -> height estimate
    m = PAT_STORIES.search(window)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 100:
                out["pdf_story_count"]    = float(n)
                mult = 10.0 if RESIDENTIAL_KW.search(window) else 12.0
                out["pdf_story_height_ft"] = clamp_height(n * mult)
        except ValueError:
            pass

    # 4. Negotiated height reduction
    m = PAT_REDUCED_TO.search(window)
    if m:
        val = clamp_height(m.group(1))
        if not np.isnan(val):
            out["pdf_reduced_to_ft"] = val

    # 5. Compatibility standard constraint height
    m = PAT_COMPAT.search(window)
    if m:
        val = clamp_height(m.group(1))
        if not np.isnan(val):
            out["pdf_compatibility_height_ft"] = val

    # 6. Staff recommendation height
    m = PAT_STAFF_HT.search(window)
    if m:
        val = clamp_height(m.group(1))
        if not np.isnan(val):
            out["pdf_staff_recommends_ht"] = val

    # 7. Staff Recommended Zoning
    m = PAT_STAFF_ZONE.search(window)
    if m:
        out["pdf_staff_recommended_zoning"] = re.sub(r"\s+", "", m.group(1).upper())
        
    # 8. Approved Zoning (Council)
    matches = PAT_ALL_ZONES.findall(window)
    if matches:
        out["pdf_approved_zoning"] = re.sub(r"\s+", "", matches[-1].upper())

    return out


def has_signal(d):
    """True if any extractable value is non-null."""
    for v in d.values():
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        return True
    return False


def filename_year(fn):
    m = re.match(r"^(\d{4})_", str(fn))
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------------

def build_case_meetings(agenda_path, label):
    """Load an agenda CSV and return case_number -> [meeting_date, ...] dict.
    Handles both lowercase (commission) and Title_Case (council) column names.
    """
    import os
    if not os.path.exists(agenda_path):
        print("  WARNING: {} not found, skipping.".format(agenda_path), flush=True)
        return {}
    ag = pd.read_csv(agenda_path, low_memory=False)
    # Normalise column names to lowercase for uniform access
    ag.columns = [c.lower() for c in ag.columns]
    # Council agendas embed description after the date — extract date portion only
    ag["meeting_date"] = ag["meeting_date"].astype(str).str.extract(
        r"(\w+ \d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})"
    )[0]
    ag["meeting_date"] = pd.to_datetime(ag["meeting_date"], errors="coerce")
    d = (
        ag.dropna(subset=["meeting_date"])
          .sort_values("meeting_date")
          .groupby("case_number")["meeting_date"]
          .apply(list)
          .to_dict()
    )
    print("  {:,} cases with {} agenda dates".format(len(d), label), flush=True)
    return d



def extract_from_corpus(trans_df, case_meetings, label):
    """Run extraction loop over a transcript dataframe. Returns list of record dicts."""
    recs = []
    n = len(trans_df)
    for row_i, row in trans_df.iterrows():
        if row_i % 2000 == 0:
            print("  [{label}] row {i:,} / {n:,}...".format(label=label, i=row_i, n=n), flush=True)
        text     = str(row.get("Raw_Text", ""))
        filename = str(row.get("Filename", ""))
        fn_yr    = filename_year(filename)
        if len(text) < 50:
            continue
        text_upper = text.upper()
        for m in CASE_PAT.finditer(text_upper):
            case_num = re.sub(r"\.0+$", "", m.group(1).upper())
            signals  = parse_window(text, m.start())
            if not has_signal(signals):
                continue
            meetings = case_meetings.get(case_num, [])
            if meetings:
                for mdate in meetings:
                    rec = {"case_number": case_num, "source_file": filename,
                           "source_date": mdate, "source_corpus": label}
                    rec.update(signals)
                    recs.append(rec)
            else:
                src = pd.Timestamp("{}-01-01".format(fn_yr)) if fn_yr else pd.NaT
                rec = {"case_number": case_num, "source_file": filename,
                       "source_date": src, "source_corpus": label}
                rec.update(signals)
                recs.append(rec)
    print("  [{label}] done. {:,} raw records.".format(len(recs), label=label), flush=True)
    return recs


# Load agendas for both corpora
print("Loading agenda date tables...", flush=True)
comm_meetings  = build_case_meetings(COMM_AGENDA,  "commission")
counc_meetings = build_case_meetings(COUNC_AGENDA, "council")

# ------------------------------------------------------------------
# Extraction: commission transcripts
# ------------------------------------------------------------------
print("\nLoading commission transcripts ({})...".format(COMM_TRANS), flush=True)
comm_trans = pd.read_csv(COMM_TRANS, low_memory=False)
print("  {:,} rows".format(len(comm_trans)), flush=True)
records = extract_from_corpus(comm_trans, comm_meetings, "commission")
del comm_trans  # free memory

# ------------------------------------------------------------------
# Extraction: council transcripts
# ------------------------------------------------------------------
print("\nLoading council transcripts ({})...".format(COUNC_TRANS), flush=True)
counc_trans = pd.read_csv(COUNC_TRANS, low_memory=False)
print("  {:,} rows".format(len(counc_trans)), flush=True)
records += extract_from_corpus(counc_trans, counc_meetings, "council")
del counc_trans

print("\nTotal raw records (both corpora): {:,}".format(len(records)), flush=True)

# ------------------------------------------------------------------
# Consolidate: one row per (case_number, source_date), first non-null wins
# ------------------------------------------------------------------
SIG_COLS = [
    "pdf_requested_zoning", "pdf_staff_recommended_zoning", "pdf_approved_zoning", "pdf_existing_zoning",
    "pdf_requested_height_ft", "pdf_requested_max_far", "pdf_proposed_height_ft",
    "pdf_story_count", "pdf_story_height_ft", "pdf_reduced_to_ft",
    "pdf_compatibility_height_ft", "pdf_staff_recommends_ht",
]

df = pd.DataFrame(records)
df["source_date"] = pd.to_datetime(df["source_date"], errors="coerce")
df = (
    df.sort_values(["case_number", "source_date"])
      .groupby(["case_number", "source_date"], as_index=False)
      .first()
      .sort_values(["case_number", "source_date"])
      .reset_index(drop=True)
)

print("\nFinal rows: {:,} | Unique cases: {:,}".format(len(df), df["case_number"].nunique()))
print("\nSignal counts (non-null rows):")
for col in SIG_COLS:
    if col in df.columns:
        n = df[col].notna().sum() if df[col].dtype != object else (df[col] != "").sum()
        print("  {:<42}: {:,}".format(col, n))

df.to_csv(OUT_PATH, index=False)
print("\nSaved: {}".format(OUT_PATH))
