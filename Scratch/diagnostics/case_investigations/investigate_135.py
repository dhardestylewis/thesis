import pandas as pd

case_num = 'C14-2018-0149'

print(f"Investigating {case_num} in Commission Transcripts...")
df_comm = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv')
comm_texts = df_comm[df_comm['Raw_Text'].str.contains(case_num, case=False, na=False)]['Raw_Text'].values
if len(comm_texts) > 0:
    text = comm_texts[0]
    idx = text.find(case_num)
    print(text[max(0, idx-200):idx+800])
else:
    print("Not found in Commission.")

print(f"\nInvestigating {case_num} in Council Transcripts...")
df_coun = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv')
coun_texts = df_coun[df_coun['Vote_Transcript'].str.contains(case_num, case=False, na=False)]['Vote_Transcript'].values
if len(coun_texts) > 0:
    text = coun_texts[0]
    idx = text.find(case_num)
    print(text[max(0, idx-200):idx+800])
else:
    print("Not found in Council.")
