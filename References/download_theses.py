import urllib.request
import re
import os
import time

dois = {
    'Gonzales_NYC311_DataDriven': '10.7916/3vcd-wd04',
    'Li_TransitDeserts_Spatial':  '10.7916/akmz-kw77',
    'Ernestus_PropertyTax_Quant': '10.7916/ax70-h904',
}

out_dir = r'C:\Users\dhl\data\Thesis\thesis\References\ComparativeTheses'
os.makedirs(out_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'text/html,application/xhtml+xml'
}

for name, doi in dois.items():
    page_url = f'https://academiccommons.columbia.edu/doi/{doi}'
    req = urllib.request.Request(page_url, headers=headers)
    try:
        html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
        # Find all links that look like downloads
        links = re.findall(r'href="([^"]+)"', html)
        dl_links = [l for l in links if 'download' in l.lower()]
        print(f'{name}: found download links: {dl_links[:5]}')
        if dl_links:
            pdf_path = dl_links[0]
            if not pdf_path.startswith('http'):
                pdf_url = 'https://academiccommons.columbia.edu' + pdf_path
            else:
                pdf_url = pdf_path
            req2 = urllib.request.Request(pdf_url, headers=headers)
            data = urllib.request.urlopen(req2, timeout=30).read()
            out_path = os.path.join(out_dir, f'{name}.pdf')
            with open(out_path, 'wb') as f:
                f.write(data)
            print(f'  -> saved {len(data)//1024}KB')
        else:
            print(f'{name}: no download link found')
    except Exception as e:
        print(f'{name}: ERROR - {e}')
    time.sleep(2)

print('Done.')
