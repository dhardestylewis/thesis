import urllib.request
import urllib.parse
import json

def search_socrata(query):
    url = f'http://api.us.socrata.com/api/catalog/v1?domains=data.austintexas.gov&q={urllib.parse.quote(query)}'
    req = urllib.request.urlopen(url)
    data = json.loads(req.read())
    for item in data['results'][:3]:
        print(f"ID: {item['resource']['id']}, Name: {item['resource']['name']}")

print('--- Floodplains ---')
search_socrata('floodplain')
print('\n--- Edwards Aquifer ---')
search_socrata('edwards aquifer recharge')
