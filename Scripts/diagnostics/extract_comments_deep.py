import fitz
import pprint

pdf_path = r'C:\Users\dhl\Downloads\HardestyLewis_Daniel_Thesis_Draft_April2026.pdf'
doc = fitz.open(pdf_path)
print(f'Total pages: {len(doc)}')

all_annots = []

for page_num, page in enumerate(doc, 1):
    # Get ALL annotation types including popups, free text, squiggly, strikeout, etc.
    for annot in page.annots():
        info = annot.info
        rect = annot.rect
        entry = {
            'page': page_num,
            'type_id': annot.type[0],
            'type': annot.type[1],
            'author': info.get('title', ''),
            'content': info.get('content', ''),
            'subject': info.get('subject', ''),
            'creation_date': info.get('creationDate', ''),
            'mod_date': info.get('modDate', ''),
            'rect': (round(rect.x0,1), round(rect.y0,1), round(rect.x1,1), round(rect.y1,1)),
        }
        all_annots.append(entry)

    # Also check raw page dict for /Annots that may be missed
    raw = page.get_text("rawdict")

print(f'\nTotal annotations (all types): {len(all_annots)}\n')
print('='*70)

for c in all_annots:
    print(f"Page {c['page']:>3} | {c['type']:<20} | Author: {c['author']}")
    if c['subject']:
        print(f"             Subject : {c['subject']}")
    if c['content']:
        print(f"             Content : {c['content']}")
    print()

# Also try to get highlighted text
print('='*70)
print('HIGHLIGHTED TEXT (extracting text under highlights):')
print('='*70)

doc2 = fitz.open(pdf_path)
for page_num, page in enumerate(doc2, 1):
    for annot in page.annots():
        if annot.type[1] in ('Highlight', 'Underline', 'StrikeOut', 'Squiggly'):
            # Get the text within the annotation rectangle
            words = page.get_text("words")
            rect = annot.rect
            # Expand rect a bit
            expanded = fitz.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2)
            highlighted_words = [w[4] for w in words if fitz.Rect(w[:4]).intersects(expanded)]
            highlighted_text = ' '.join(highlighted_words)
            info = annot.info
            print(f"Page {page_num} | {annot.type[1]} | Author: {info.get('title','')}")
            print(f"  Highlighted text: \"{highlighted_text}\"")
            if info.get('content'):
                print(f"  Comment: {info.get('content')}")
            print()
