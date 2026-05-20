import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv", low_memory=False)
cols = ["cumulative_unofficial_protest_intensity", "Requested_max_height_ft", "median_household_income", "longitude", "latitude", "market_value", "building_age"]
df = df.dropna(subset=cols)
if len(df) > 10000:
    df = df.sample(10000, random_state=42)

y = df["cumulative_unofficial_protest_intensity"].values
T = df["Requested_max_height_ft"].values
income = df["median_household_income"].values
market_value = df["market_value"].values
building_age = df["building_age"].values

from sklearn.preprocessing import StandardScaler
spatial_scaled = StandardScaler().fit_transform(np.column_stack([df["longitude"].values, df["latitude"].values]))
income_scaled = StandardScaler().fit_transform(income.reshape(-1, 1)).flatten()

X_confounded = np.column_stack([spatial_scaled, income_scaled, market_value, building_age])
q_conf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X_confounded, y).predict(X_confounded)
p_conf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X_confounded, T).predict(X_confounded)
cate_conf = (y - q_conf) / (T - p_conf + 1e-6)

X_inv = np.column_stack([market_value, building_age])
q_inv = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X_inv, y).predict(X_inv)
p_inv = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X_inv, T).predict(X_inv)
cate_idml = (y - q_inv) / (T - p_inv + 1e-6)

mask_conf = (cate_conf > np.percentile(cate_conf, 5)) & (cate_conf < np.percentile(cate_conf, 95))
mask_idml = (cate_idml > np.percentile(cate_idml, 5)) & (cate_idml < np.percentile(cate_idml, 95))

# Calculate stats
slope_conf, intercept_conf, r_value_conf, p_value_conf, std_err_conf = stats.linregress(income[mask_conf], cate_conf[mask_conf])
slope_idml, intercept_idml, r_value_idml, p_value_idml, std_err_idml = stats.linregress(income[mask_idml], cate_idml[mask_idml])

income_std = np.std(income)

print("=== Standard DML ===")
print(f"P-value: {p_value_conf:.6f}")
print(f"Effect per 1 Std Dev ($40k) Income increase: {slope_conf * income_std:.5f} increase in CATE")

print("\n=== I-DML ===")
print(f"P-value: {p_value_idml:.6f}")
print(f"Effect per 1 Std Dev ($40k) Income increase: {slope_idml * income_std:.5f} increase in CATE")

