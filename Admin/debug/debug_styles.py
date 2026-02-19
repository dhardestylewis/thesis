from docx import Document

doc = Document(r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Updates\Templates\Weekly or Biweekly Status Update Tempate.docx")
print("Styles found:")
for s in doc.styles:
    print(f" - '{s.name}'")
