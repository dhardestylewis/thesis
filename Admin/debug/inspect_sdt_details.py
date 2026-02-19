from docx import Document

def inspect_sdt_details(filepath):
    doc = Document(filepath)
    print(f"Inspecting: {filepath}")
    
    # Iterate through all elements in the body to find SDTs
    for element in doc.element.body:
        xml = element.xml
        if "w:sdt" in xml:
            print("--- Found SDT ---")
            # print a snippet to identify it
            if "w:dropDownList" in xml:
                print("Type: Drop-Down List")
                # Try to extract list items
                # They look like <w:listItem w:displayText="Yes" w:value="Yes"/>
                import re
                items = re.findall(r'<w:listItem w:displayText="(.*?)"', xml)
                print(f"Options: {items}")
            else:
                print("Type: Other SDT (Date, Text, etc)")
                
if __name__ == "__main__":
    path = r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Updates\Templates\Weekly or Biweekly Status Update Tempate.docx"
    inspect_sdt_details(path)
