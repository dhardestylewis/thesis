"""Merge Census tract timeseries onto panel with leakage-safe forward-fill."""
import csv, sys, os
csv.field_size_limit(min(sys.maxsize, 2**31-1))

PANEL_PATH = "Data/Panel/Output/Property_Year_Panel.csv"
CENSUS_PATH = "Data/Panel/Intermediate/census_tract_timeseries.csv"
OUT_PATH = "Data/Panel/Intermediate/panel_with_census.csv"

print("=== Census Tract Timeseries Merge ===")

# Load census data: (geoid, vintage) -> row
census = {}
census_cols = None
vintages_available = set()

with open(CENSUS_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    census_cols = [c for c in reader.fieldnames if c not in ("geoid", "vintage")]
    for row in reader:
        geoid = row["geoid"].strip()
        vintage = int(row["vintage"])
        vintages_available.add(vintage)
        census[(geoid, vintage)] = {c: row[c] for c in census_cols}

sorted_vintages = sorted(vintages_available)
print("Census vintages: %s" % sorted_vintages)
print("Census variables: %s" % census_cols)
print("Census (geoid, vintage) entries: %d" % len(census))

def get_census_forward_fill(geoid, year):
    """Get census data for nearest vintage <= year (no leakage)."""
    for v in reversed(sorted_vintages):
        if v <= year:
            rec = census.get((geoid, v))
            if rec:
                return rec, str(v)
    return None, ""

# Merge onto panel
total = 0
matched = 0
no_geoid = 0

with open(PANEL_PATH, "r", encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    out_fields = reader.fieldnames + ["census_" + c for c in census_cols] + ["census_vintage"]

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            total += 1
            geoid = row.get("nearby_GEOID", "").strip()
            year = int(row["year"])
            out_row = dict(row)

            if not geoid:
                no_geoid += 1
                for c in census_cols:
                    out_row["census_" + c] = ""
                out_row["census_vintage"] = ""
            else:
                rec, vintage_used = get_census_forward_fill(geoid, year)
                if rec:
                    for c in census_cols:
                        out_row["census_" + c] = rec[c]
                    out_row["census_vintage"] = vintage_used
                    matched += 1
                else:
                    for c in census_cols:
                        out_row["census_" + c] = ""
                    out_row["census_vintage"] = ""

            writer.writerow(out_row)

            if total % 1000000 == 0:
                print("  %d rows..." % total)

print("Total rows: %d" % total)
print("Census matched: %d (%.1f%%)" % (matched, 100*matched/total))
print("No GEOID: %d" % no_geoid)
print("Unmatched (have GEOID but no census): %d" % (total - matched - no_geoid))
print("Output: %s" % OUT_PATH)
