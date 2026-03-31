import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT_DIR = r"C:/Users/dhl/data/thesis/thesis"
DATA_FILE = os.path.join(ROOT_DIR, "Data/Panel/Output/Property_Year_Panel_v3.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Analysis/Output/Track1_Predictive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def define_archetypes(prop_type):
    # Synthetic categorical map representing Stage B Project Scope from historical TCAD
    if pd.isna(prop_type): return "Minor_Infill"
    pt = str(prop_type).upper()
    if "MULTIFAMILY" in pt or "B" in pt: return "Multifamily"
    if "COMMERCIAL" in pt or "F" in pt: return "Commercial_Conversion"
    if "PUD" in pt: return "PUD_Major_Project"
    return "Minor_Infill"

def execute_stage_b():
    print("\n--- Stage B: Project Type & Scale Architecture ---")
    print("Loading TCAD Panel Array...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    print("Mapping Conditional Archetypes (Model 2: What kind of project?)...")
    df["project_archetype"] = df["property_type_code"].apply(define_archetypes)
    print(df["project_archetype"].value_counts())
    
    # 4 core spatial/feasibility proxies for forecasting capability without data leakage
    features = ["improvement_market_value", "land_market_value", "deed_acreage", "year_built"]
    model_df = df.dropna(subset=features).sample(min(200000, len(df)), random_state=42)
    X = model_df[features]
    y = model_df["project_archetype"]
    
    print("\nTraining Gradient Boosted Stage B Classifier...")
    cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, loss_function="MultiClass", verbose=50)
    cb.fit(X, y)
    
    preds = cb.predict(X).flatten()
    
    print("\n--- Stage B Forecast Synthesis ---")
    f1 = f1_score(y, preds, average="macro")
    print(f"Macro-F1 (Project Class): {f1:.4f}")
    print(f"Overall Accuracy: {accuracy_score(y, preds):.4f}")
    print("Classification Report:")
    print(classification_report(y, preds))
    
    with open(os.path.join(OUTPUT_DIR, "StageB_Results.txt"), "w") as f:
        f.write(f"Macro-F1 (Project Class): {f1:.4f}\n")
        f.write(f"Overall Accuracy: {accuracy_score(y, preds):.4f}\n")
    print("Exported Stage B baseline metrics!\n")

if __name__ == "__main__":
    execute_stage_b()
