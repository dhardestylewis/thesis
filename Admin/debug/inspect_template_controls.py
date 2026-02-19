from docx import Document
from docx.oxml.ns import qn

def inspect_docx(filepath):
    doc = Document(filepath)
    print(f"Inspecting: {filepath}")
    
    # SDT are often not directly exposed in python-docx API, usually need valid OXML access.
    # We will search the XML of the document body for 'w:sdt' or 'w:ddList'
    
    xml = doc.element.xml
    if "w:sdt" in xml:
        print("Found Structured Document Tags (SDT) - likely content controls.")
    else:
        print("No SDT found in main body.")
        
    if "w:ddList" in xml:
        print("Found Drop-Down List elements.")
    else:
        print("No Drop-Down List elements found.")

    # Check for text like "Choose an item." which is common default text
    if "Choose an item." in xml:
        print("Found default placeholder text 'Choose an item.'")

if __name__ == "__main__":
    path = r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Updates\Templates\Weekly or Biweekly Status Update Tempate.docx"
    inspect_docx(path)
