import glob, json, re, sys, os
import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

def extract_all(html_file, out_csv):
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    data_match = re.search(r'Plotly\.newPlot.*?,\s*(\[\{.*?\}\])\s*,', content)
    
    records = []
    if data_match:
        data = json.loads(data_match.group(1))
        
        for trace in data:
            name = trace.get('name', '')
            era = name.replace('≤', '').strip()
            if not era: continue
            
            x = trace['x']
            y = trace['y']
            z = trace['z']
            
            for i, pct in enumerate(y):
                for j, period in enumerate(x):
                    records.append({
                        'era_cutoff': era,
                        'period_seq': period,
                        'petition_pct': pct,
                        'z_outcome': z[i][j]
                    })
    else:
        # Fallback
        return False
        
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return True

def main():
    targets = ['survival', 'downgrade', 'commission', 'council']
    for t in targets:
        f = os.path.join(OUT_DIR, f'causal_lstm_biweekly_{t}.html')
        out_csv = os.path.join(OUT_DIR, f'master_surfaces_{t}.csv')
        if os.path.exists(f):
            print(f'Extracting all frames from {t}... ', end='')
            success = extract_all(f, out_csv)
            print(success)
        else:
            print(f'File not found: {f}')

if __name__ == '__main__':
    main()
