import glob, json, re, sys

def extract(html_file, out_csv):
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    data_match = re.search(r'Plotly\.newPlot.*?,\s*(\[\{.*?\}\])\s*,', content)
    
    if data_match:
        try:
            data = json.loads(data_match.group(1))
            trace = data[-1]
            x = trace['x']
            y = trace['y']
            z = trace['z']
            
            with open(out_csv, 'w', encoding='utf-8') as f:
                f.write('period_seq,petition_pct,z_outcome\n')
                for i, pct in enumerate(y):
                    for j, period in enumerate(x):
                        f.write(f'{period},{pct},{z[i][j]}\n')
            return True
        except:
            pass
            
    x_match = [m.group(1) for m in re.finditer(r'\"x\":(\[[0-9\.,]+\])', content)]
    y_match = [m.group(1) for m in re.finditer(r'\"y\":(\[[0-9\.,]+\])', content)]
    z_match = [m.group(1) for m in re.finditer(r'\"z\":(\[\[.*?\]\])', content)]
    
    if x_match and y_match and z_match:
        x = json.loads(x_match[-1])
        y = json.loads(y_match[-1])
        z = json.loads(z_match[-1])
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write('period_seq,petition_pct,z_outcome\n')
            for i, pct in enumerate(y):
                for j, period in enumerate(x):
                    f.write(f'{period},{pct},{z[i][j]}\n')
        return True
    return False

for f in glob.glob(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\causal_lstm_biweekly_*.html'):
    out_csv = f.replace('.html', '_raw_surface.csv')
    success = extract(f, out_csv)
    print(f'Extracted {f}: {success}')
