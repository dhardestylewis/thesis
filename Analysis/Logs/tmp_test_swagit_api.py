import requests
import json

swagit_id = '45757'
url = f"https://austintx.swagit.com/api/v1/agenda/{swagit_id}"
headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get(url, headers=headers)

if r.status_code == 200:
    data = r.json()
    with open("swagit_api.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Success! Wrote swagit_api.json")
else:
    print(f"Failed: {r.status_code} {r.text}")
