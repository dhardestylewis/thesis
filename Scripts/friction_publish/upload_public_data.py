"""Upload the two objects the public page needs into austin-friction-public."""
import boto3, threading, sys
from boto3.s3.transfer import TransferConfig
from pathlib import Path

BUCKET = 'austin-friction-public'
DATA = Path(r"C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases")
FILES = [
    (DATA / "austin_base_geometries_cached.fgb", "austin_base_geometries.fgb", "application/octet-stream"),
    (DATA / "austin_friction_grid.f32", "austin_friction_grid.f32", "application/octet-stream"),
]

env = {}
for line in Path(r"C:\Users\dhl\data\Projects\Properlytic_UI\v0-properlytic-8v\.env.local").read_text(encoding='utf-8', errors='ignore').splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip().strip('"').strip("'")

acct = env.get('R2_ACCOUNT_ID') or env.get('CLOUDFLARE_ACCOUNT_ID')
s3 = boto3.client('s3', endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
                  aws_access_key_id=env['R2_ACCESS_KEY_ID'],
                  aws_secret_access_key=env['R2_SECRET_ACCESS_KEY'], region_name='auto')

cfg = TransferConfig(multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2,
                     max_concurrency=4, use_threads=True)

class Progress:
    def __init__(self, name, total):
        self.name, self.total, self.seen, self.lock, self.mark = name, total, 0, threading.Lock(), 0
    def __call__(self, n):
        with self.lock:
            self.seen += n
            pct = 100 * self.seen / self.total
            if pct - self.mark >= 10:
                self.mark = pct
                print(f"  {self.name}: {pct:.0f}% ({self.seen/1e6:.0f}/{self.total/1e6:.0f} MB)", flush=True)

for path, key, ctype in FILES:
    size = path.stat().st_size
    print(f"uploading {path.name} -> s3://{BUCKET}/{key} ({size/1e6:.0f} MB)", flush=True)
    s3.upload_file(str(path), BUCKET, key, ExtraArgs={'ContentType': ctype},
                   Config=cfg, Callback=Progress(key, size))
    head = s3.head_object(Bucket=BUCKET, Key=key)
    ok = head['ContentLength'] == size
    print(f"  done: {head['ContentLength']:,} bytes {'(matches local)' if ok else '(SIZE MISMATCH)'}", flush=True)
    if not ok:
        sys.exit(1)
print("upload complete", flush=True)
