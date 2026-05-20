import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv", low_memory=False)
cols = ["cumulative_unofficial_protest_intensity", "Requested_max_height_ft", "median_household_income", 
        "council_district", "latitude", "longitude", "market_value", "building_age"]
df = df.dropna(subset=cols)
if len(df) > 10000:
    df = df.sample(10000, random_state=42)

y = df["cumulative_unofficial_protest_intensity"].values
T = df["Requested_max_height_ft"].values
income = df["median_household_income"].values
spatial_x = df["longitude"].values
spatial_y = df["latitude"].values
market_value = df["market_value"].values
building_age = df["building_age"].values

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
scaler_inc = StandardScaler()
income_scaled = scaler_inc.fit_transform(income.reshape(-1, 1)).flatten()
scaler_spat = StandardScaler()
spatial_scaled = scaler_spat.fit_transform(np.column_stack([spatial_x, spatial_y]))

print("=== I-DML Metrics ===")
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

slope_conf = np.polyfit(income[mask_conf], cate_conf[mask_conf], 1)[0]
slope_idml = np.polyfit(income[mask_idml], cate_idml[mask_idml], 1)[0]

print(f"Standard DML CATE vs Income Slope: {slope_conf:.10f}")
print(f"I-DML CATE vs Income Slope:        {slope_idml:.10f}")


print("\n=== Adversarial Metrics ===")
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * 0.5

class AdversarialSpatialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
        self.predictor = nn.Linear(2, 1)
        self.discriminator = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, c):
        z = self.encoder(c)
        pred_y = self.predictor(z)
        z_rev = GradReverse.apply(z)
        pred_inc = self.discriminator(z_rev)
        return z, pred_y, pred_inc

adv_model = AdversarialSpatialNet()
opt_adv = optim.Adam(adv_model.parameters(), lr=0.01)

C_t = torch.tensor(spatial_scaled, dtype=torch.float32)
inc_t = torch.tensor(income_scaled, dtype=torch.float32).view(-1, 1)
y_t = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

for epoch in range(150):
    opt_adv.zero_grad()
    z, pred_y, pred_inc = adv_model(C_t)
    loss = nn.MSELoss()(pred_y, y_t) + nn.MSELoss()(pred_inc, inc_t)
    loss.backward()
    opt_adv.step()

with torch.no_grad():
    z_final, pred_y_final, pred_inc_final = adv_model(C_t)
    pred_y_np = pred_y_final.numpy().flatten()
    pred_inc_np = pred_inc_final.numpy().flatten()

r2_y_adv = r2_score(y_scaled, pred_y_np)
r2_inc_adv = r2_score(income_scaled, pred_inc_np)

pca_z = PCA(n_components=2).fit_transform(spatial_scaled)
pca_pred_y = LinearRegression().fit(pca_z, y_scaled).predict(pca_z)
pca_pred_inc = LinearRegression().fit(pca_z, income_scaled).predict(pca_z)
r2_y_pca = r2_score(y_scaled, pca_pred_y)
r2_inc_pca = r2_score(income_scaled, pca_pred_inc)

print(f"PCA R^2 (Protest):         {r2_y_pca:.4f}")
print(f"PCA R^2 (Income):          {r2_inc_pca:.4f}")
print(f"Adversarial R^2 (Protest): {r2_y_adv:.4f}")
print(f"Adversarial R^2 (Income):  {r2_inc_adv:.4f}")
