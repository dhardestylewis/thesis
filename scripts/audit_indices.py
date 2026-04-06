import re

filepath = 'Thesis_Draft/Draft_v1/Austin_NIMBY_Thesis_Draft.tex'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

print("--- Subsections ---")
subsections = re.findall(r'\\subsection\{(.*?)\}', text)
for sub in subsections:
    print(sub)

print("\n--- Sections ---")
sections = re.findall(r'\\section\{(.*?)\}', text)
for sec in sections:
    print(sec)

print("\n--- Table Captions ---")
# Only captions inside table environments
table_blocks = re.findall(r'\\begin\{table\}(.*?)\\end\{table\}', text, re.DOTALL)
for block in table_blocks:
    cap = re.search(r'\\caption\{(.*?)\}', block)
    if cap:
        print(cap.group(1))

