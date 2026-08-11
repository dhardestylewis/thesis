"""Verify the uploaded objects byte-for-byte at the offsets the page will request."""
import boto3, numpy as np
from pathlib import Path

BUCKET = 'austin-friction-public'
DATA = Path(r"C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases")
SLICE = 271567 * 3 * 4

env = {}
for line in Path(r"C:\Users\dhl\data\Projects\Properlytic_UI\v0-properlytic-8v\.env.local").read_text(encoding='utf-8', errors='ignore').splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k, _, v = line.partition('='); env[k.strip()] = v.strip().strip('"').strip("'")
acct = env.get('R2_ACCOUNT_ID') or env.get('CLOUDFLARE_ACCOUNT_ID')
s3 = boto3.client('s3', endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
                  aws_access_key_id=env['R2_ACCESS_KEY_ID'],
                  aws_secret_access_key=env['R2_SECRET_ACCESS_KEY'], region_name='auto')

for key, local in [('austin_base_geometries.fgb', 'austin_base_geometries_cached.fgb'),
                   ('austin_friction_grid.f32', 'austin_friction_grid.f32')]:
    remote = s3.head_object(Bucket=BUCKET, Key=key)['ContentLength']
    print(f"{key:28s} remote {remote:,}  local {(DATA/local).stat().st_size:,}  "
          f"{'match' if remote == (DATA/local).stat().st_size else 'MISMATCH'}")

# The exact range the page issues for the default 45 ft scenario
tier = 45 - 5
start = tier * SLICE
body = s3.get_object(Bucket=BUCKET, Key='austin_friction_grid.f32',
                     Range=f'bytes={start}-{start + SLICE - 1}')['Body'].read()
remote_tier = np.frombuffer(body, dtype=np.float32)
local_tier = np.fromfile(DATA / 'austin_friction_grid.f32', dtype=np.float32,
                         count=271567 * 3, offset=start)
print(f"45 ft tier: {len(body):,} bytes, byte-exact vs local: {np.array_equal(remote_tier, local_tier)}")
print(f"  sanity at dose 0.21 -> mean delay {np.clip(remote_tier[0::3] * 0.21, -365, 3650).mean():.0f} days")
