import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import json

chunks = []
for chunk in pd.read_csv(r"C:\Users\dhl\data\thesis\thesis\Data\Panel\Output\Property_Year_Panel_v3.csv", usecols=["property_type_code", "improvement_market_value", "land_market_value", "deed_acreage", "year_built"], chunksize=250000, low_memory=False):
    chunks.append(chunk.sample(frac=0.02, random_state=42))

df = pd.concat(chunks)
def define_archetypes(prop_type):
    if pd.isna(prop_type): return "Minor_Infill"
    pt = str(prop_type).upper()
    if "MULTIFAMILY" in pt or "B" in pt: return "Multifamily"
    if "COMMERCIAL" in pt or "F" in pt: return "Commercial_Conversion"
    if "PUD" in pt: return "PUD_Major_Project"
    return "Minor_Infill"

df["project_archetype"] = df["property_type_code"].apply(define_archetypes)
features = ["improvement_market_value", "land_market_value", "deed_acreage", "year_built"]
model_df = df.dropna(subset=features)
X = model_df[features]
y = model_df["project_archetype"]

cb = CatBoostClassifier(iterations=25, depth=4, learning_rate=0.05, loss_function="MultiClass", verbose=0)
cb.fit(X, y)
rep = classification_report(y, cb.predict(X).flatten(), zero_division=0, output_dict=True)

with open('C:/Users/dhl/data/thesis/thesis/Analysis/Scripts/Modeling/report.json', 'w') as f:
    json.dump(rep, f, indent=4)
