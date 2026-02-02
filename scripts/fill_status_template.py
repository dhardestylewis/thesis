from docx import Document
from docx.shared import Pt
from docx.oxml.shared import OxmlElement, qn
import os
import re

def add_hyperlink(paragraph, text, url):
    """
    Adds a hyperlink to a paragraph using OXML.
    """
    # Create the w:hyperlink tag
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    # Create the w:r element
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    
    # Blue color
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "0563C1")
    rPr.append(c)
    
    # Underline
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)

    new_run.append(rPr)
    new_run.text = text
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return hyperlink

def process_line_with_links(paragraph, line_text, bold_prefix=None):
    """
    Parses line_text for Markdown links [Text](URL) and adds runs/hyperlinks to paragraph.
    If bold_prefix is provided (e.g. 'Key:'), it is added as a bold run first.
    """
    if bold_prefix:
        paragraph.add_run(bold_prefix).bold = True
    
    # Pattern: [Text](URL)
    parts = re.split(r'(\[.*?\]\(.*?\))', line_text)
    
    for part in parts:
        link_match = re.match(r'\[(.*?)\]\((.*?)\)', part)
        if link_match:
            text = link_match.group(1)
            url = link_match.group(2)
            add_hyperlink(paragraph, text, url)
        else:
            if part:
                 # Check for raw http links if needed, otherwise just add text
                 paragraph.add_run(part)

def process_text_file(txt_path):
    """Parses the text file into metadata, sections, and table data."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]

    data = {
        "metadata": {},
        "sections": {"A": [], "B": [], "C": []},
        "table": [],
        "footer": []
    }
    
    current_section = None
    in_table = False
    in_footer = False

    for line in lines:
        if not line: continue
        
        # Metadata
        if line.startswith("Name:"): data["metadata"]["Name"] = line
        elif line.startswith("Project Name:"): data["metadata"]["Project"] = line
        elif line.startswith("Date:"): data["metadata"]["Date"] = line
        
        # Sections
        elif line.lower().startswith("section a:"): current_section = "A"
        elif line.lower().startswith("section b:"): current_section = "B"
        elif line.lower().startswith("section c:"): current_section = "C"
        elif line.lower().startswith("section d:"): 
            current_section = "D"
            in_table = True
        
        # Footer
        elif line.startswith("PROGRESS LINK:"):
            in_footer = True
            current_section = None
            data["footer"].append(line)
        
        elif in_footer:
            data["footer"].append(line)

        # Content Collection
        elif current_section in ["A", "B", "C"]:
            data["sections"][current_section].append(line)
            
        elif current_section == "D" and in_table and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if parts[0] == "Challenge": continue # Skip header
            data["table"].append(parts)

    return data

def fill_template(template_path, output_path, data):
    doc = Document(template_path)
    
    # 1. Update Metadata
    for p in doc.paragraphs[:10]:
        text = p.text.strip()
        if text.startswith("Name:"):
            p.text = data["metadata"].get("Name", text)
            p.runs[0].bold = True
        elif text.startswith("Project Name:"):
            p.text = data["metadata"].get("Project", text)
            p.runs[0].bold = True
        elif text.startswith("Date:"):
            p.text = data["metadata"].get("Date", text)
            p.runs[0].bold = True

    # 2. Update Sections A, B, C
    def clear_section_content(doc, section_header, unique_next_headers):
        """
        Removes all paragraphs between section_header and any of the next_headers.
        """
        start_clearing = False
        paras_to_delete = []
        
        for p in doc.paragraphs:
            text = p.text.strip().lower()
            
            # Check if we hit the start header
            if text.startswith(section_header.lower()):
                start_clearing = True
                continue # Don't delete the header itself
            
            # Check if we hit the next section
            for nh in unique_next_headers:
                if text.startswith(nh.lower()):
                    start_clearing = False
                    break
            
            if start_clearing:
                # We are in the section. Delete this paragraph.
                # However, maybe we want to keep empty lines? 
                # Template usually has " * Item 1..." placeholders.
                paras_to_delete.append(p)
        
        for p in paras_to_delete:
            p._element.getparent().remove(p._element)

    # Clear placeholders first
    clear_section_content(doc, "Section A:", ["Section B:"])
    clear_section_content(doc, "Section B:", ["Section C:"])
    clear_section_content(doc, "Section C:", ["Section D:"])

    def insert_lines_after_header(header_prefix_list, lines):
        target_p = None
        for p in doc.paragraphs:
            for hp in header_prefix_list:
                if p.text.strip().lower().startswith(hp.lower()):
                    target_p = p
                    break
            if target_p: break
        
        if target_p:
            # Use manual bullet formatting since 'List Bullet' style is missing
            current_p = target_p
            for line in lines:
                new_p = doc.add_paragraph(style='normal') 
                p_fmt = new_p.paragraph_format
                p_fmt.left_indent = Pt(18) 
                p_fmt.first_line_indent = Pt(-18) 
                
                # Format text
                # Try to emulate bullet?
                # new_p.add_run("• ") # We need to handle this carefully with the bold/link logic
                
                # We need a run for the bullet first
                new_p.add_run("• ")

                if "**" in line:
                    clean_line = line.replace("* ", "").replace("- ", "")
                    # Split only on the first colon to separate Key from Value
                    parts = clean_line.split(":", 1)
                    if len(parts) > 1:
                        # parts[0] is the key (e.g. "**Data Collection**")
                        key_text = parts[0].replace("**", "") + ":"
                        process_line_with_links(new_p, parts[1], bold_prefix=key_text)
                    else:
                        # No colon
                        process_line_with_links(new_p, clean_line.replace("**", ""), bold_prefix=None) 
                else:
                    process_line_with_links(new_p, line.replace("* ", "").replace("- ", ""))
                
                # Move this new_p to right position
                new_p._element.getparent().remove(new_p._element)
                current_p._element.addnext(new_p._element)
                current_p = new_p # Advance

    insert_lines_after_header(["Section A:"], data["sections"]["A"])
    insert_lines_after_header(["Section B:"], data["sections"]["B"])
    insert_lines_after_header(["Section C:"], data["sections"]["C"])

    # 3. Update Table (Section D)
    if doc.tables:
        table = doc.tables[0]
        raw_rows = data["table"]
        
        for i, row_data in enumerate(raw_rows):
            table_idx = i + 1
            if table_idx < len(table.rows):
                row = table.rows[table_idx]
                # Clear existing and add fresh run?
                row.cells[0].text = ""
                process_line_with_links(row.cells[0].paragraphs[0], row_data[0])
                
                row.cells[1].text = ""
                process_line_with_links(row.cells[1].paragraphs[0], row_data[1])
                # Dropdown in cells[2] left alone
            else:
                pass # Ran out of rows
        
        # Delete unused template rows
        start_del = len(raw_rows) + 1
        while len(table.rows) > start_del:
            row = table.rows[-1]
            row._element.getparent().remove(row._element)

    # 4. Footer (Progress Link)
    doc.add_paragraph("_" * 30)
    for line in data["footer"]:
        p = doc.add_paragraph()
        process_line_with_links(p, line)

    doc.save(output_path)
    print(f"Filled Template Saved: {output_path}")

if __name__ == "__main__":
    txt_source = r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Updates\2026-02-01_Status_Update_GDOC_READY.txt"
    tpl_source = r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Updates\Templates\Weekly or Biweekly Status Update Tempate.docx"
    output = r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Updates\Columbia_Thesis_Status_Update_Daniel_Hardesty_Lewis_2026-02-01_Predicting_NIMBYism.docx"
    
    parsed = process_text_file(txt_source)
    fill_template(tpl_source, output, parsed)
