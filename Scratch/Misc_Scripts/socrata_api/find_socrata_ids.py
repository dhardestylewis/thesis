import urllib.request
import urllib.parse
import json

domains = 'data.austintexas.gov'
queries = ['Plan Review Cases', 'site_plan_case', 'Issued Building Permits']

for q in queries:
    url = f'http://api.us.socrata.com/api/catalog/v1?domains={domains}&search_context={domains}&q={urllib.parse.quote(q)}'
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read())
        print(f'\nResults for {q}:')
        for item in data['results'][:3]:
            print(f"  - {item['resource']['name']} -> {item['resource']['id']}")
    except Exception as e:
        print(f'Error for {q}: {e}')
