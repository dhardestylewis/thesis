import urllib.request
import os

urls = [
    ("https://services.austintexas.gov/edims/document.cfm?id=414950", "20230613_Minutes_HPC.pdf"),
    ("https://services.austintexas.gov/edims/document.cfm?id=409725", "20230613_Draft_Minutes_HPC.pdf"),
    ("https://services.austintexas.gov/edims/document.cfm?id=294911", "20180327_Draft_Minutes_HPC.pdf"),
    ("https://services.austintexas.gov/edims/document.cfm?id=299896", "20180327_Minutes_HPC.pdf")
]

headers = {"User-Agent": "Mozilla/5.0"}
output_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Housing_PDFs"

for url, filename in urls:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as f:
                f.write(content)
            print(f"Success: {file_path}")
    except Exception as e:
        print(f"Error: {e}")
