import pandas as pd
import re

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv')
pattern = re.compile(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})')

count = 0
for t in df['Raw_Text'].dropna():
    m = pattern.search(t.upper())
    if m:
        idx = m.start()
        print('\n--- EXAMPLE ---')
        print(t[max(0,idx-50):min(len(t),idx+200)])
        count += 1
        if count > 5:
            break
