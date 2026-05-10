import pandas as pd
nlp = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\interim\nlp_event_log.csv", low_memory=False)
print("Total rows:", len(nlp))
print("Unique cases:", nlp["case_number"].nunique())
print("Source values:", nlp["source"].unique())
print("Date range:", nlp["event_date"].min(), "->", nlp["event_date"].max())
print("Nonzero oppose:", nlp["oppose"].gt(0).sum(), "traffic:", nlp["traffic"].gt(0).sum(), "density:", nlp["density"].gt(0).sum())
print(nlp.sort_values("oppose", ascending=False).head(3).to_string())
