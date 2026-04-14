import re
import sys
from collections import defaultdict
from datetime import datetime
import os

def main():
    path = "Thesis_Draft/Draft_v1/Austin_NIMBY_Thesis_Draft.git-blame.txt"
    if not os.path.exists(path):
        print(f"Blame file not found: {path}")
        return

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    paragraphs = []
    
    for line in lines:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
        if not match:
            continue
            
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Get content after ") "
        content_match = re.search(r'\d+\)\s(.*)', line)
        if not content_match:
            continue
            
        content = content_match.group(1).strip()
        
        # Ignore lines that are short or just latex commands
        if len(content) < 50 or content.startswith('\\'):
            continue
            
        paragraphs.append({
            'date': date_obj,
            'date_str': date_str,
            'content': content
        })

    paragraphs.sort(key=lambda x: x['date'])
    
    with open("RECENCY_AUDIT_2026.md", "w", encoding='utf-8') as out:
        out.write("# Comprehensive Recency Audit — Git-History Based\n\n")
        out.write(f"Scanned {len(paragraphs)} paragraph-level text blocks.\n\n")
        
        out.write("## Stale Paragraphs (Needs Review against Pipeline output)\n\n")
        
        oldest = paragraphs[:20] # top 20 oldest
        for i, p in enumerate(oldest, 1):
            out.write(f"### {i}. Date: {p['date_str']}\n")
            out.write(f"> {p['content'][:300]}...\n\n")
            
    print("Recency audit complete. Output written to RECENCY_AUDIT_2026.md")

if __name__ == '__main__':
    main()
