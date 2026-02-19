"""Write audit results to a UTF-8 file."""
import csv, sys, os
csv.field_size_limit(min(sys.maxsize, 2**31-1))

out = open("Data/Panel/Logs/audit_results.txt", "w", encoding="utf-8")

def p(s):
    print(s)
    out.write(s + "\n")

# --- OVERLAP AUDIT ---
p("=" * 70)
p("EARS vs LUI VARIABLE OVERLAP AUDIT")
p("=" * 70)

LUI_PATH = "Data/Zoning_Cases/Source_Data/land_use_inventory_prefetched.csv"
EARS_PATH = "Data/Panel/Intermediate/ears_2024_clean.csv"

# LUI columns
with open(LUI_PATH, "r") as f:
    lui_cols = sorted(csv.DictReader(f).fieldnames)
p("LUI columns: %s" % lui_cols)

# Load LUI data
lui_data = {}
with open(LUI_PATH, "r") as f:
    for row in csv.DictReader(f):
        pid = row.get("parcel_id_10", "").strip()
        if pid:
            lui_data[pid] = row
p("LUI parcels: %d" % len(lui_data))

# Load EARS 2024
ears_all = {}
with open(EARS_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row.get("_parcel_id_10", "").strip()
        if pid:
            ears_all[pid] = row
p("EARS 2024 parcels: %d" % len(ears_all))

overlap_pids = list(set(ears_all.keys()) & set(lui_data.keys()))
p("Overlap parcels: %d" % len(overlap_pids))

# Sample comparison
p("")
p("--- 5 sample parcels ---")
for pid in overlap_pids[:5]:
    lui = lui_data[pid]
    ears = ears_all[pid]
    p("Parcel %s:" % pid)
    p("  LUI.land_use = %s" % lui.get("land_use", ""))
    p("  EARS.land_use_code = %s" % ears.get("land_use_code", ""))
    p("  LUI.general_land_use = %s" % lui.get("general_land_use", ""))
    p("  EARS.property_category_code = %s" % ears.get("property_category_code", ""))
    p("  EARS.subcategory_code = %s" % ears.get("subcategory_code", ""))
    p("  EARS.zoning_code = %s" % ears.get("zoning_code", ""))
    p("  LUI.shape_area = %s" % str(lui.get("shape_area", ""))[:20])
    p("  EARS.deed_acreage = %s" % ears.get("deed_acreage", ""))
    p("")

# Systematic check 
p("--- Systematic overlap (5000 parcels) ---")
compare_fields = [
    ("EARS.land_use_code", "land_use_code", "LUI.land_use", "land_use"),
    ("EARS.property_category_code", "property_category_code", "LUI.general_land_use", "general_land_use"),
]
for ears_label, ears_col, lui_label, lui_col in compare_fields:
    same = diff = ears_empty = lui_empty = 0
    sample_diffs = []
    for pid in overlap_pids[:5000]:
        ev = ears_all[pid].get(ears_col, "").strip()
        lv = lui_data[pid].get(lui_col, "").strip()
        if not ev: ears_empty += 1
        elif not lv: lui_empty += 1
        elif ev == lv: same += 1
        else:
            diff += 1
            if len(sample_diffs) < 3:
                sample_diffs.append((pid, ev, lv))
    p("%s vs %s: same=%d diff=%d ears_empty=%d lui_empty=%d" % (
        ears_label, lui_label, same, diff, ears_empty, lui_empty))
    for pid, ev, lv in sample_diffs:
        p("  Example: %s -> EARS=%s LUI=%s" % (pid, ev, lv))

# ---- COVERAGE AUDIT ----
p("")
p("=" * 70)
p("PANEL COVERAGE AUDIT")
p("=" * 70)

f2 = open("Data/Panel/Output/Property_Year_Panel.csv", "r", encoding="utf-8")
reader = csv.DictReader(f2)
cols = reader.fieldnames
counts = {c: 0 for c in cols}
total = 0
ears_src = {}

for row in reader:
    total += 1
    for c in cols:
        if row.get(c, "").strip():
            counts[c] += 1
    src = row.get("ears_source", "").strip()
    ears_src[src] = ears_src.get(src, 0) + 1
f2.close()

p("Total rows: %d" % total)
p("Columns: %d" % len(cols))
p("")
p("EARS source distribution:")
for src, cnt in sorted(ears_src.items(), key=lambda x: -x[1]):
    label = src if src else "(empty/unmatched)"
    p("  %s: %d (%.1f%%)" % (label, cnt, 100*cnt/total))

p("")
p("Column coverage (sorted by coverage):")
coverage = sorted([(c, counts[c], 100*counts[c]/total) for c in cols], key=lambda x: x[2])
for c, cnt, pct in coverage:
    p("  %s: %d (%.1f%%)" % (c, cnt, pct))

out.close()
print("Done. Results written to Data/Panel/Logs/audit_results.txt")
