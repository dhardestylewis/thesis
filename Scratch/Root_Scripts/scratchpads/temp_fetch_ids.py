import urllib.request
import json

datasets = {
    'Land Use Inventory Detailed': 'https://data.austintexas.gov/api/views?q=Land%20Use%20Inventory%20Detailed',
    'Future Land Use': 'https://data.austintexas.gov/api/views?q=Future%20Land%20Use',
    'Boundaries Jurisdictions': 'https://data.austintexas.gov/api/views?q=BOUNDARIES_jurisdictions'
}

for name, url in datasets.items():
    print(f'\n--- {name} ---')
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            for item in data[:2]:
                print(f"ID: {item['id']} | Name: {item['name']}")
    except Exception as e:
        print(f"Error: {e}")
