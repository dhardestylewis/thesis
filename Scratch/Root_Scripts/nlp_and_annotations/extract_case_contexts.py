import pandas as pd
import re
from pathlib import Path

def extract_contexts():
    df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Scratch\annotated_case_studies.csv')
    cases = ['C14-2013-0098', 'C14-2013-0104', 'C14-2008-0247']
    
    out_path = Path(r'c:\Users\dhl\data\Thesis\thesis\Scratch\case_study_contexts.md')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# Case Study Transcripts - Relevant Contexts\n\n")
        
        for case in cases:
            f.write(f"## Case: {case}\n\n")
            
            # Find all transcripts containing this case
            subset = df[df['Raw_Text'].str.contains(case, na=False)]
            
            if len(subset) == 0:
                f.write("*No transcripts found.*\n\n")
                continue
                
            for idx, row in subset.iterrows():
                filename = row['Filename']
                text = str(row['Raw_Text'])
                
                # Find all occurrences of the case in the text
                starts = [m.start() for m in re.finditer(case, text)]
                
                if not starts:
                    continue
                    
                f.write(f"### File: {filename}\n")
                
                for i, start in enumerate(starts):
                    # Extract 1500 chars before and after
                    window_start = max(0, start - 1500)
                    window_end = min(len(text), start + 2500)
                    
                    context = text[window_start:window_end]
                    # Clean up the text a bit for readability
                    context = re.sub(r'\s+', ' ', context).strip()
                    
                    f.write(f"**Occurrence {i+1}:**\n")
                    f.write(f"> ... {context} ...\n\n")
            
            f.write("---\n\n")
            
    print(f"Extraction complete! Saved to {out_path}")

if __name__ == "__main__":
    extract_contexts()
