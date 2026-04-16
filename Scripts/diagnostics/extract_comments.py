import fitz  # PyMuPDF

pdf_path = r'C:\Users\dhl\Downloads\HardestyLewis_Daniel_Thesis_Draft_April2026.pdf'
doc = fitz.open(pdf_path)
print(f'Total pages: {len(doc)}')

comments = []
for page_num, page in enumerate(doc, 1):
    for annot in page.annots():
        info = annot.info
        comments.append({
            'page': page_num,
            'type': annot.type[1],
            'author': info.get('title', 'Unknown'),
            'content': info.get('content', ''),
            'subject': info.get('subject', ''),
        })

print(f'Total annotations found: {len(comments)}\n')
for c in comments:
    print(f"--- Page {c['page']} | Type: {c['type']} | Author: {c['author']}")
    if c['subject']:
        print(f"    Subject: {c['subject']}")
    if c['content']:
        print(f"    Comment: {c['content']}")
    print()
