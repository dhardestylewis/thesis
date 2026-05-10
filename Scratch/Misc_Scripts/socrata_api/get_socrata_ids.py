import urllib.request
import re

url = 'https://data.austintexas.gov/browse?q=Plan%20Review%20Cases'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode()
    matches = re.findall(r'href="/dataset/.*?/([a-z0-9]{4}-[a-z0-9]{4})"', html)
    matches_building = re.findall(r'href="/Building-and-Development/.*?/([a-z0-9]{4}-[a-z0-9]{4})"', html)
    print('Found dataset IDs:', set(matches + matches_building))
except Exception as e:
    print('Error:', e)
