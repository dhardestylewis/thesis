import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

PANEL_PATH = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
OUT_DIR = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\06_HBU_Prospectivity"
MODEL_OUT_PATH = os.path.join(OUT_DIR, "hbu_generator_weights.json")

FEATURES = [
    "appraised_value",
    "land_acres",
    "building_age",
    "median_household_income",
    "renter_share",
    "total_population",
    "latitude",
    "longitude",
    "council_district"
]
TARGET = "pdf_requested_height_ft"

def main():
    print(f"Loading panel data from {PANEL_PATH}...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    # Filter to first period to get static case-level snapshot
    df_static = df[df["period_seq"] == 1].copy()
    
    # Filter to cases that actually have a requested height. Let XGBoost handle missing features natively.
    df_valid = df_static.dropna(subset=[TARGET]).copy()
    
    print(f"Found {len(df_valid)} valid zoning cases with requested height and required features.")
    
    X = df_valid[FEATURES]
    y = df_valid[TARGET]
    
    print("Splitting 80/20 train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    print("\n--- Evaluation Metrics ---")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Mean Absolute Error (MAE): {mae:.2f} feet")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} feet")
    
    print(f"\nSaving model to {MODEL_OUT_PATH}...")
    model.save_model(MODEL_OUT_PATH)
    print("Done! Highest and Best Use (HBU) ML Generator is fully trained and exported.")

if __name__ == "__main__":
    main()
