import pandas as pd
import re

df_comm = pd.read_csv('Data/interim/commission_transcripts.csv')

zone_regex_old = r'\b(?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-?[0-9A-Z]+){0,4}\b'
pattern_req_to_old = re.compile(r'(?i)request.{0,40}?(' + zone_regex_old + r').{0,30}?\bto\b.{0,30}?(' + zone_regex_old + r')')

c = 0
for text in df_comm['Raw_Text'].dropna():
    c += len(pattern_req_to_old.findall(text.upper()))
print('OLD Regex Request Matches:', c)

zone_regex_new = r'\b(?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:[0-9]+[A-Z]*)?(?:-[A-Z0-9]+)*\b'
pattern_req_to_new = re.compile(r'(?i)request.{0,40}?(' + zone_regex_new + r').{0,30}?\bto\b.{0,30}?(' + zone_regex_new + r')')

c2 = 0
for text in df_comm['Raw_Text'].dropna():
    c2 += len(pattern_req_to_new.findall(text.upper()))
print('NEW Regex Request Matches:', c2)
