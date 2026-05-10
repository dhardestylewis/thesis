import pandas as pd
import re

case_pattern = re.compile(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4}(?:\.[A-Z0-9]+)?)')

case_num = 'C14-2018-0149'
df_comm = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv')

for i, row in df_comm.iterrows():
    text = row['Raw_Text']
    if pd.isna(text): continue
    text_str = str(text).upper()
    
    for m in case_pattern.finditer(text_str):
        case = m.group(1)
        if case == case_num:
            idx = m.start()
            start = max(0, idx - 50)
            end = min(len(text_str), idx + 800)
            
            forward_window = text_str[idx:end]
            next_match = case_pattern.search(forward_window[len(case):])
            if next_match:
                end = idx + len(case) + next_match.start()
            
            window = text_str[start:end]
            print("--- WINDOW ---")
            print(window)
            print("--------------")
