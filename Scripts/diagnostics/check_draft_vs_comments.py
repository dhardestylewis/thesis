import fitz
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

pdf_workspace = r'C:\Users\dhl\data\Thesis\thesis\HardestyLewis_Daniel_Thesis_Draft_April2026.pdf'
doc = fitz.open(pdf_workspace)
print(f'Workspace PDF pages: {len(doc)}')

# Pages of interest from advisor comments
# Note: page numbers may have shifted since draft grew 51->74 pages
# Check pages around 21 and 38 as well as a wider range
pages_to_check = [21, 22, 38, 39, 40, 41, 42]

for pn in pages_to_check:
    if pn > len(doc):
        break
    page = doc[pn - 1]
    text = page.get_text().strip()
    print(f'\n{"="*70}')
    print(f'PAGE {pn}:')
    print(text[:3000])
