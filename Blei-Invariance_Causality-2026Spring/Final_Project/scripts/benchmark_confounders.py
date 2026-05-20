import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv", low_memory=False)
df = df.dropna(subset=["cumulative_unofficial_protest_intensity", "Requested_max_height_ft", "median_household_income", "market_value", "building_age", "longitude", "latitude"])

# Target and Treatment
y = df["cumulative_unofficial_protest_intensity"].values
T = df["Requested_max_height_ft"].values

# Standardize to get comparable effect sizes (Cohen's d equivalent)
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
T_scaled = scaler.fit_transform(T.reshape(-1, 1)).flatten()

# Observed covariates
covariates = {
    "Median Income": df["median_household_income"].values,
    "Market Value": df["market_value"].values,
    "Building Age": df["building_age"].values,
    "Spatial X": df["longitude"].values,
    "Spatial Y": df["latitude"].values
}

print("=== Benchmarking Observed Covariates against E-Value (1.49) ===")
print("VanderWeele (2017) Continuous RR approximation: exp(0.91 * d)\n")

for name, cov in covariates.items():
    if cov.std() == 0: continue
    cov_scaled = scaler.fit_transform(cov.reshape(-1, 1)).flatten()
    
    # Association with Outcome (Protests)
    reg_y = LinearRegression().fit(cov_scaled.reshape(-1, 1), y_scaled)
    effect_y = abs(reg_y.coef_[0])
    rr_y = np.exp(0.91 * effect_y)
    
    # Association with Treatment (Upzoning Height)
    reg_t = LinearRegression().fit(cov_scaled.reshape(-1, 1), T_scaled)
    effect_t = abs(reg_t.coef_[0])
    rr_t = np.exp(0.91 * effect_t)
    
    print(f"[{name}]")
    print(f"  -> RR with Protests: {rr_y:.3f}")
    print(f"  -> RR with Height:   {rr_t:.3f}")
    print(f"  -> Max Association:  {max(rr_y, rr_t):.3f}")
    if max(rr_y, rr_t) >= 1.49:
        print(f"     *** SURPASSES E-VALUE (1.49) ***")
    print()
