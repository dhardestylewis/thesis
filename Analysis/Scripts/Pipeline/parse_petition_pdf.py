"""
Parse Petition PDF — Full Dual-Format Position-Based Parser
============================================================
Handles:
- Format A (early pages): row# at x~82, TCAD(dash) at x~130, owner at x~227
- Format B (newer pages): TCAD(10-digit) at x~53, Signature(yes/no) at x~484
- Continuation pages: no case header, carries case# from previous page
- Header variant: "# TCAD ID Address" at top of continuation pages
"""
import pdfplumber
import csv
import re
import os
from collections import defaultdict

PDF_PATH = "Data/Documents/C241282.PD.NRN.petitions.pdf"
OUT_CSV = "Data/Protest_Petitions/petition_signers_from_pdf.csv"
OUT_SUMMARY = "Data/Protest_Petitions/petition_summary_from_pdf.csv"

TCAD_DASH = re.compile(r"^\d{2}-\d{4}-\d{4}")
TCAD_LONG = re.compile(r"^\d{10,14}$")
CASE_RE = re.compile(r"(C\d+[A-Z]*-(?:(?:19|20)\d{2}|\d{2})-\d+(?:\.\d+\w*)?(?:\([A-Z]+\))?)")


def normalize_tcad(tid):
    return tid.replace("-", "").replace(" ", "").lstrip("0")


def group_words_to_rows(words, y_tolerance=3):
    rows = defaultdict(list)
    for w in words:
        y_key = round(w["top"] / y_tolerance) * y_tolerance
        rows[y_key].append(w)
    return [(y, sorted(rows[y], key=lambda w: w["x0"])) for y in sorted(rows)]


def is_tcad(text):
    return bool(TCAD_DASH.match(text) or TCAD_LONG.match(text))


def detect_format(word_rows):
    """Detect page format. Returns 'A', 'B', or None."""
    # Check for signature/petition header (Format B)
    for y, ws in word_rows:
        texts = [w["text"].lower() for w in ws]
        full = " ".join(texts)
        if "signature" in full and ("petition" in full or "precent" in full or "area" in full):
            return "B"
    # Check for continuation header: "# TCAD ID Address"
    for y, ws in word_rows[:5]:
        texts = [w["text"].lower() for w in ws]
        full = " ".join(texts)
        if "tcad" in full and "address" in full:
            return "B"
    # Format A: row# at x < 100, TCAD at x 100-230
    for y, ws in word_rows:
        if len(ws) >= 2:
            w0, w1 = ws[0], ws[1]
            if w0["x0"] < 100 and w1["x0"] < 230:
                try:
                    int(w0["text"])
                    if is_tcad(w1["text"]):
                        return "A"
                except ValueError:
                    pass
    # Fallback: if page has TCADs at x < 150, treat as format B
    for y, ws in word_rows:
        for w in ws:
            if is_tcad(w["text"]) and w["x0"] < 150:
                return "B"
    return None


def extract_case_and_date(word_rows):
    """Extract case number and date from header."""
    case_number = None
    date_str = ""
    for y, ws in word_rows[:12]:
        text = " ".join(w["text"] for w in ws)
        m = CASE_RE.search(text)
        if m:
            case_number = m.group(1)
        if "Date:" in text:
            idx = text.index("Date:")
            date_str = text[idx + 5:].strip()
    return case_number, date_str


def extract_year(case_number, date_str):
    year = None
    if case_number:
        m_yr = re.search(r"((?:19|20)\d\d)", case_number)
        if m_yr:
            year = int(m_yr.group(1))
    if not year and date_str:
        m_yr2 = re.search(r"((?:19|20)\d\d)", date_str)
        if m_yr2:
            year = int(m_yr2.group(1))
    return year


def parse_format_a(word_rows, case_number, year, date_str):
    """Old format: row# at x<100, TCAD at x~130, owner at x~227, area, pct."""
    parcels = []
    for y, ws in word_rows:
        if not ws or ws[0]["x0"] > 100:
            continue
        try:
            row_num = int(ws[0]["text"])
        except ValueError:
            continue
        if row_num < 1 or row_num > 60:
            continue

        tcad = None
        area = None
        pct = None
        owner_words = []

        for w in ws[1:]:
            x, text = w["x0"], w["text"]
            if is_tcad(text) and x < 230:
                tcad = text
            elif "%" in text and x > 450:
                try:
                    pct = float(text.replace("%", ""))
                except ValueError:
                    pass
            elif x > 350 and x < 470 and "%" not in text:
                try:
                    area = float(text.replace(",", ""))
                except ValueError:
                    pass
            elif 200 < x < 370:
                owner_words.append(text)

        if pct == 0 and tcad is None:
            continue

        if tcad is not None:
            parcels.append({
                "case_number": case_number or "",
                "year": year or "",
                "date": date_str,
                "tcad_id": tcad,
                "tcad_normalized": normalize_tcad(tcad),
                "owner_name": " ".join(owner_words),
                "area_sqft": area or 0,
                "area_pct": pct or 0,
                "signed": "1" if (pct is not None and pct > 0) else "0",
            })
    return parcels


def parse_format_b(word_rows, case_number, year, date_str):
    """New format: TCAD at x<150. Just scan every row for a TCAD ID."""
    parcels = []

    for y, ws in word_rows:
        if not ws:
            continue

        tcad = None
        signature = None
        area = None
        pct = None
        owner_words = []

        for w in ws:
            x, text = w["x0"], w["text"]

            if is_tcad(text) and x < 150:
                tcad = text
            elif text.lower() in ("yes", "no") and x > 400:
                signature = text.lower()
            elif "%" in text and x > 400:
                try:
                    pct = float(text.replace("%", ""))
                except ValueError:
                    pass
            elif 200 < x < 420:
                owner_words.append(text)

        if tcad is not None:
            if signature == "yes":
                is_signed = True
            elif signature == "no":
                is_signed = False
            else:
                is_signed = pct is not None and pct > 0

            parcels.append({
                "case_number": case_number or "",
                "year": year or "",
                "date": date_str,
                "tcad_id": tcad,
                "tcad_normalized": normalize_tcad(tcad),
                "owner_name": " ".join(owner_words),
                "area_sqft": area or 0,
                "area_pct": pct or 0,
                "signed": "1" if is_signed else "0",
            })
    return parcels


def parse_pdf():
    print("Parsing PDF: %s" % PDF_PATH)

    all_rows = []
    case_stats = defaultdict(lambda: {"total_parcels": 0, "signers": 0, "year": None})
    format_counts = {"A": 0, "B": 0, "carry": 0, "skip": 0}

    # Track the last known case number for continuation pages
    last_case = None
    last_year = None
    last_date = ""

    with pdfplumber.open(PDF_PATH) as pdf:
        print("  Total pages: %d" % len(pdf.pages))

        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words()
            if not words:
                format_counts["skip"] += 1
                continue

            word_rows = group_words_to_rows(words)

            # Try to extract case number
            case_number, date_str = extract_case_and_date(word_rows)

            if case_number:
                year = extract_year(case_number, date_str)
                last_case = case_number
                last_year = year
                last_date = date_str
            else:
                # Continuation page — use last known case
                case_number = last_case
                year = last_year
                date_str = last_date
                if case_number:
                    format_counts["carry"] += 1

            if not case_number:
                format_counts["skip"] += 1
                continue

            # Detect format and parse
            fmt = detect_format(word_rows)
            if fmt == "A":
                parcels = parse_format_a(word_rows, case_number, year, date_str)
                format_counts["A"] += 1
            elif fmt == "B":
                parcels = parse_format_b(word_rows, case_number, year, date_str)
                format_counts["B"] += 1
            else:
                # Try both — B first (more common), then A
                parcels = parse_format_b(word_rows, case_number, year, date_str)
                if not parcels:
                    parcels = parse_format_a(word_rows, case_number, year, date_str)
                if parcels:
                    format_counts["B"] += 1
                else:
                    format_counts["skip"] += 1

            all_rows.extend(parcels)
            for p in parcels:
                case_stats[case_number]["total_parcels"] += 1
                case_stats[case_number]["year"] = year
                if p["signed"] == "1":
                    case_stats[case_number]["signers"] += 1

    print("  Format counts: %s" % format_counts)
    print("  Total parcel rows: %d" % len(all_rows))
    print("  Unique cases: %d" % len(case_stats))
    total_signers = sum(1 for r in all_rows if r["signed"] == "1")
    print("  Total signers: %d" % total_signers)
    print("  Total non-signers: %d" % (len(all_rows) - total_signers))

    # Write detailed CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["case_number", "year", "date", "tcad_id",
                  "tcad_normalized", "owner_name", "area_sqft", "area_pct", "signed"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print("\n  Detailed CSV: %s (%d rows)" % (OUT_CSV, len(all_rows)))

    # Write summary
    summary_fields = ["case_number", "year", "total_parcels", "signers", "signer_pct"]
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for cn in sorted(case_stats):
            s = case_stats[cn]
            writer.writerow({
                "case_number": cn,
                "year": s["year"] or "",
                "total_parcels": s["total_parcels"],
                "signers": s["signers"],
                "signer_pct": round(100 * s["signers"] / max(s["total_parcels"], 1), 1),
            })
    print("  Summary CSV: %s (%d cases)" % (OUT_SUMMARY, len(case_stats)))

    # Per-year summary
    year_stats = defaultdict(lambda: {"cases": 0, "total_signers": 0, "total_parcels": 0})
    for cn, s in case_stats.items():
        yr = s.get("year", 0) or 0
        year_stats[yr]["cases"] += 1
        year_stats[yr]["total_signers"] += s["signers"]
        year_stats[yr]["total_parcels"] += s["total_parcels"]

    print("\n=== PDF signers per year ===")
    print("Year | Cases | Total parcels | Signers | Avg signers/case")
    print("-----|-------|---------------|---------|------------------")
    for yr in sorted(year_stats):
        if yr == 0:
            continue
        s = year_stats[yr]
        avg = s["total_signers"] / max(s["cases"], 1)
        print("%d | %5d | %13d | %7d | %.1f" % (
            yr, s["cases"], s["total_parcels"], s["total_signers"], avg))

    total_c = sum(s["cases"] for yr, s in year_stats.items() if yr > 0)
    total_s = sum(s["total_signers"] for yr, s in year_stats.items() if yr > 0)
    total_p = sum(s["total_parcels"] for yr, s in year_stats.items() if yr > 0)
    print("Tot  | %5d | %13d | %7d | %.1f" % (total_c, total_p, total_s, total_s / max(total_c, 1)))


if __name__ == "__main__":
    parse_pdf()
