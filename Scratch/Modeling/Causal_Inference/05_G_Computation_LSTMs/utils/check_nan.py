import json
with open(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\causal_lstm_monthly_overlay_downgrade.html', 'r', encoding='utf-8') as f: content = f.read()

import re
match = re.search(r'\"z\":(\[\[.*?\]\])', content)
if match:
    z_str = match.group(1)
    if 'null' in z_str or 'NaN' in z_str:
        print('Z matrix contains nulls or NaNs!')
    else:
        print('Z matrix is valid floats.')
        # Print first few elements to verify scale
        try:
            z_arr = json.loads(z_str)
            print('Z[0][:5]:', z_arr[0][:5])
        except Exception as e:
            print('JSON load failed', e)
