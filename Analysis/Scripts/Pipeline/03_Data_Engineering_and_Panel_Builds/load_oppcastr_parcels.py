#!/usr/bin/env python3
"""
Load LUI 2024 parcel geometries into oppcastr via REST API.
Reads the_geom WKT from the CSV and sends in batches of 50 via RPC.
"""
import csv
import json
import sys
import time
import urllib.request

csv.field_size_limit(10**8)

BASE_URL = "https://lzwuerruoiqdoiycvntf.supabase.co/rest/v1/rpc"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx6d3VlcnJ1b2lxZG9peWN2bnRmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MjEzMDQ5NSwiZXhwIjoyMDg3NzA2NDk1fQ.x0bhMiH9SwbzXpGejVsDYps-8MbKmQVvNyDXNgtUsHM"
HEADERS = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

LUI_PATH = "Data/CoA_Open_Data/Land_Use/LUI_2024_7vsm-dvxg.csv"
BATCH_SIZE = 50  # small batches because WKT geometries are huge
FUNC = "oppcastr_bulk_insert_parcels"

def send_batch(records):
    payload = json.dumps({"data": records}).encode()
    url = f"{BASE_URL}/{FUNC}"
    req = urllib.request.Request(url, data=payload, headers=HEADERS, method="POST")
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                err = e.read().decode() if hasattr(e, "read") else str(e)
                print(f"  FAILED: {err[:200]}")
                return 0

total = 0
skipped = 0
batch = []
start = time.time()

with open(LUI_PATH, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        geom_wkt = row.get("the_geom", "").strip()
        prop_id = row.get("PROPERTY_ID", "").strip()
        if not geom_wkt or not prop_id:
            skipped += 1
            continue

        batch.append({"acct": prop_id, "prop_id": prop_id, "wkt": geom_wkt})
        if len(batch) >= BATCH_SIZE:
            n = send_batch(batch)
            total += n
            batch = []
            if total % 1000 == 0:
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                remaining = (284958 - total) / rate / 60 if rate > 0 else 0
                print(f"  {total} parcels loaded ({rate:.0f}/s, ~{remaining:.0f} min remaining)", flush=True)

if batch:
    n = send_batch(batch)
    total += n

elapsed = time.time() - start
print(f"\nDone! Loaded {total} parcels, skipped {skipped} ({elapsed:.0f}s)")
