import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

df = pd.read_csv(r"C:\Users\dhl\data\thesis\thesis\Data\Panel\Output\Property_Year_Panel_Enriched.csv", low_memory=False, usecols=['property_type_code', 'improvement_market_value', 'land_market_value', 'deed_acreage', 'year_built'])

def define_archetypes(prop_type):
    if pd.isna(prop_type): return "Minor_Infill"
    pt = str(prop_type).upper()
    if "MULTIFAMILY" in pt or "B" in pt: return "Multifamily"
    if "COMMERCIAL" in pt or "F" in pt: return "Commercial_Conversion"
    if "PUD" in pt: return "PUD_Major_Project"
    return "Minor_Infill"

df["project_archetype"] = df["property_type_code"].apply(define_archetypes)
features = ["improvement_market_value", "land_market_value", "deed_acreage", "year_built"]
model_df = df.dropna(subset=features).sample(200000, random_state=42)
X = model_df[features]
y = model_df["project_archetype"]

cb = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05, loss_function="MultiClass", verbose=0)
cb.fit(X, y)
print(classification_report(y, cb.predict(X).flatten(), zero_division=0))
