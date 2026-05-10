import re
with open(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\causal_lstm_monthly_overlay_downgrade.html', 'r', encoding='utf-8') as f: content = f.read()
types = set(re.findall(r'"type":"(.*?)"', content))
print("Trace types:", types)
if "surface" in content:
    print("SURFACE EXISTS!")
else:
    print("SURFACE IS MISSING!")
