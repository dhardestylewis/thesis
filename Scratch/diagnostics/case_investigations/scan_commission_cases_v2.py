import pandas as pd
import re
import time

print("Loading massive transcript CSV...")
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")

print("Loading master zoning cases...")
df_model = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv")

cases_to_find = set(df_model['Core_Case'].dropna().unique())

print(f"Searching for {len(cases_to_find)} unique cases in the commission transcripts...")

start = time.time()
giant_string = " ".join(df_comm['Filename'].fillna('')) + " " + " ".join(df_comm['Raw_Text'].astype(str).fillna(''))
giant_string = giant_string.upper()

pattern = r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})'
found_cases = set(re.findall(pattern, giant_string))

matched_cases = cases_to_find.intersection(found_cases)

df_found = pd.DataFrame({'Core_Case': list(matched_cases)})
df_found.to_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_reached_cases.csv", index=False)
print(f"Successfully matched {len(matched_cases)} cases that reached the Commission in {time.time() - start:.2f} seconds!")
