"""Create a dedicated public bucket for the Austin friction map data.

Isolated on purpose: enabling public access on properlytic-raw-data would expose
every pipeline object in it. This bucket holds only the two files the page needs.
"""
import boto3, botocore, json, urllib.request
from pathlib import Path

BUCKET = 'austin-friction-public'

env = {}
for line in Path(r"C:\Users\dhl\data\Projects\Properlytic_UI\v0-properlytic-8v\.env.local").read_text(encoding='utf-8', errors='ignore').splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip().strip('"').strip("'")

acct = env.get('R2_ACCOUNT_ID') or env.get('CLOUDFLARE_ACCOUNT_ID')
s3 = boto3.client('s3', endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
                  aws_access_key_id=env['R2_ACCESS_KEY_ID'],
                  aws_secret_access_key=env['R2_SECRET_ACCESS_KEY'], region_name='auto')

existing = [b['Name'] for b in s3.list_buckets()['Buckets']]
if BUCKET in existing:
    print(f"bucket {BUCKET} already exists")
else:
    s3.create_bucket(Bucket=BUCKET)
    print(f"created bucket {BUCKET}")

# CORS so the page on worldcastr.com may issue Range requests
s3.put_bucket_cors(Bucket=BUCKET, CORSConfiguration={'CORSRules': [{
    'AllowedOrigins': ['https://worldcastr.com', 'https://www.worldcastr.com',
                       'http://localhost:8000', 'http://127.0.0.1:8000'],
    'AllowedMethods': ['GET', 'HEAD'],
    'AllowedHeaders': ['Range', 'Content-Type'],
    'ExposeHeaders': ['Content-Range', 'Content-Length', 'Accept-Ranges'],
    'MaxAgeSeconds': 86400,
}]})
print('CORS rules applied (GET/HEAD + Range from worldcastr.com and localhost)')

# Try to turn on the managed public URL; needs an R2-admin scoped API token
token = env.get('CLOUDFLARE_API_TOKEN')
req = urllib.request.Request(
    f"https://api.cloudflare.com/client/v4/accounts/{acct}/r2/buckets/{BUCKET}/domains/managed",
    data=json.dumps({'enabled': True}).encode(),
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}, method='PUT')
try:
    with urllib.request.urlopen(req, timeout=25) as r:
        print('public dev URL enabled:', json.load(r).get('result'))
except urllib.error.HTTPError as e:
    print(f'could not enable public URL automatically: HTTP {e.code} {e.read().decode()[:160]}')
    print('  -> needs a one-time toggle in the Cloudflare dashboard, or an R2-admin token')
