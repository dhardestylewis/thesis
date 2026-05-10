import pandas as pd

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df_model = pd.read_csv(model_csv)
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")

missing_cases = set(df_model[df_model['Requested_Zoning'].isna()]['Core_Case'].dropna().unique())

with open(r"c:\Users\dhl\data\Thesis\thesis\Scratch\missing_zones_output.txt", "w") as f:
    f.write(f"Total missing cases: {len(missing_cases)}\n\n")
    
    count = 0
    for t in df_comm['Raw_Text'].dropna():
        text_str = str(t).upper()
        for case in list(missing_cases):
            idx = text_str.find(case)
            if idx != -1:
                f.write(f"\n--- MISSING CASE: {case} ---\n")
                start = max(0, idx - 100)
                end = min(len(text_str), idx + 600)
                f.write(text_str[start:end] + "\n")
                count += 1
                missing_cases.remove(case)
                break
                
        if count >= 30:
            break
