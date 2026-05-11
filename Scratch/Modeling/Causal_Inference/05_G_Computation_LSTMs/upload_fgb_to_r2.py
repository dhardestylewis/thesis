"""
upload_fgb_to_r2.py  —  uploads austin_base_geometries.fgb to Cloudflare R2
and prints the public URL.
"""
import boto3
import os
from pathlib import Path

R2_ACCOUNT_ID    = "7f58e07bff423d2120acf10aa6bf7a32"
R2_ACCESS_KEY_ID = "a9ad4cd53aaf0193e35f4b2b48edbad5"
R2_SECRET        = "7a4eba060df826a5ba1f3a293017ab2207cc4a39536d7472880b7e2776c7f981"
BUCKET           = "properlytic-raw-data"
OBJECT_KEY       = "public/austin_base_geometries.fgb"

FGB_PATH = Path(r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\austin_base_geometries.fgb")

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
)

print(f"Uploading {FGB_PATH.name} ({FGB_PATH.stat().st_size / 1024**2:.1f} MB) to R2...")

s3.upload_file(
    str(FGB_PATH),
    BUCKET,
    OBJECT_KEY,
    ExtraArgs={"ContentType": "application/octet-stream"},
    Callback=lambda n: print(f"  {n / 1024**2:.1f} MB uploaded", end="\r", flush=True),
)

PUBLIC_URL = f"https://pub-{R2_ACCOUNT_ID}.r2.dev/{OBJECT_KEY}"
print(f"\nDone! Public URL:\n  {PUBLIC_URL}")
