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
CACHE_KEY        = "public/inference_cache.npy"

FGB_PATH   = Path(r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\austin_base_geometries.fgb")
CACHE_PATH = Path(r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\inference_cache.npy")

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
)

def upload(local_path, key):
    print(f"Uploading {local_path.name} ({local_path.stat().st_size / 1024**2:.1f} MB) to R2...")
    s3.upload_file(
        str(local_path),
        BUCKET,
        key,
        ExtraArgs={"ContentType": "application/octet-stream"},
        Callback=lambda n: print(f"  {n / 1024**2:.1f} MB uploaded", end="\r", flush=True),
    )
    print(f"\nDone: https://pub-{R2_ACCOUNT_ID}.r2.dev/{key}")

upload(FGB_PATH, OBJECT_KEY)
if CACHE_PATH.exists():
    upload(CACHE_PATH, CACHE_KEY)
else:
    print(f"Skipping {CACHE_PATH.name} (not found yet)")
