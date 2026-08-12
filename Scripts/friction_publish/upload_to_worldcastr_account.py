"""Upload the friction map data into the R2 bucket owned by the worldcastr.com account.

The original bucket was created in a different Cloudflare account than the one
holding the worldcastr.com zone, and an R2 custom domain requires both to live in
the same account. This re-uploads the two objects into 465fb..., where
data.worldcastr.com can actually be attached to them.

Credentials come from a private file outside the repo and are never printed.
"""
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from pathlib import Path
import sys

ACCOUNT = '465fb49b30af5adbba4bf08bcf12b5ce'   # Daniel@homecastr.com's Account, owns worldcastr.com
BUCKET = 'austin-friction-public'
SECRETS = Path(r"C:\Users\dhl\.codex\secrets\r2-worldcastr.env")
DATA = Path(r"C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases")

OBJECTS = [
    ('austin_base_geometries.fgb', 'austin_base_geometries_cached.fgb'),
    ('austin_friction_grid.f32', 'austin_friction_grid.f32'),
]

env = {}
for line in SECRETS.read_text(encoding='utf-8', errors='ignore').splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip().strip('"').strip("'")

missing = [k for k in ('R2W_ACCESS_KEY_ID', 'R2W_SECRET_ACCESS_KEY') if not env.get(k)]
if missing:
    sys.exit(f"missing keys in {SECRETS}: {', '.join(missing)}")

s3 = boto3.client(
    's3', endpoint_url=f"https://{ACCOUNT}.r2.cloudflarestorage.com",
    aws_access_key_id=env['R2W_ACCESS_KEY_ID'],
    aws_secret_access_key=env['R2W_SECRET_ACCESS_KEY'],
    region_name='auto',
    # R2 rejects the newer default integrity headers boto3 sends
    config=Config(request_checksum_calculation='when_required',
                  response_checksum_validation='when_required'),
)

# R2 multipart parts must be uniform; 64 MiB keeps the 361 MiB grid to six parts.
cfg = TransferConfig(multipart_threshold=200 * 1024 * 1024,
                     multipart_chunksize=64 * 1024 * 1024,
                     max_concurrency=4)

for key, local_name in OBJECTS:
    local = DATA / local_name
    size = local.stat().st_size
    try:
        if s3.head_object(Bucket=BUCKET, Key=key)['ContentLength'] == size:
            print(f"{key:28s} already present at {size:,} bytes, skipping")
            continue
        print(f"{key:28s} present but wrong size, re-uploading")
    except s3.exceptions.ClientError:
        pass

    print(f"{key:28s} uploading {size:,} bytes...", flush=True)
    s3.upload_file(str(local), BUCKET, key, Config=cfg,
                   ExtraArgs={'ContentType': 'application/octet-stream'})
    remote = s3.head_object(Bucket=BUCKET, Key=key)['ContentLength']
    print(f"{key:28s} uploaded, remote {remote:,} {'match' if remote == size else 'MISMATCH'}")

print("\nApplying CORS so the page may issue Range requests...")
s3.put_bucket_cors(Bucket=BUCKET, CORSConfiguration={'CORSRules': [{
    'AllowedOrigins': ['https://worldcastr.com', 'https://www.worldcastr.com',
                       'http://localhost:8000', 'http://127.0.0.1:8000'],
    'AllowedMethods': ['GET', 'HEAD'],
    'AllowedHeaders': ['Range', 'Content-Type'],
    'ExposeHeaders': ['Content-Range', 'Content-Length', 'Accept-Ranges'],
    'MaxAgeSeconds': 86400,
}]})
print("CORS applied (GET/HEAD + Range from worldcastr.com and localhost)")
