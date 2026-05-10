import pandas as pd
df = pd.read_csv('Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv', low_memory=False)
first_petition = df[df['petition_event'] == 1].groupby('case_number')['period_seq'].min()
df['first_petition_seq'] = df['case_number'].map(first_petition)
df = df[(df['first_petition_seq'].isna()) | (df['period_seq'] <= df['first_petition_seq'])]

ann = df.sort_values(["case_number", "period_seq"]).groupby(["case_number", "year"]).last().reset_index()
events = df.groupby(["case_number", "year"])["petition_event"].max().reset_index()
ann = ann.drop(columns=["petition_event"]).merge(events, on=["case_number", "year"], how="left")

print("Ann petitions:", ann['petition_event'].sum())

# build horizons
for window in [1, 3, 5]:
    evt = ann[["case_number", "year", "petition_event"]].copy()
    evt["future_event"] = evt.groupby("case_number")["petition_event"].transform(
        lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1]
    )
    print(f"Window {window} targets:", evt['future_event'].sum())
