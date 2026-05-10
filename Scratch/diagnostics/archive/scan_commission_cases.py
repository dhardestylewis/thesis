import pandas as pd
import re

print("Loading massive transcript CSV...")
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")

print("Loading master zoning cases...")
df_model = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv")

unique_cases = set()

# Pattern for Austin zoning cases: C14-2021-0001, NPA-2020-0015.01, etc.
# We will just compile a list of all cases in the model and search for them in the text
cases_to_find = df_model['Core_Case'].dropna().unique()

print(f"Searching for {len(cases_to_find)} unique cases in the commission transcripts...")

# To make this fast, we can concatenate all text into one giant string
# and then just check if case in giant_string.
giant_string = " ".join(df_comm['Filename'].fillna('')) + " " + " ".join(df_comm['Raw_Text'].astype(str).fillna(''))
giant_string = giant_string.upper()

found_cases = []
for case in cases_to_find:
    if str(case).upper() in giant_string:
        found_cases.append(case)

df_found = pd.DataFrame({'Core_Case': found_cases})
df_found.to_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_reached_cases.csv", index=False)
print(f"Successfully matched {len(found_cases)} cases that reached the Commission!")
