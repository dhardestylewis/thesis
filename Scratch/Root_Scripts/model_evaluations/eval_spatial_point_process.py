import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ZONING_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
SPATIAL_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\Panel\spatial_allocation_panel.csv"
OUT_PLOT   = r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts\causal_spatial_intensity_2023.png"

# Hyperparameters
GRID_SIZE = 50
EPOCHS = 1000
LR = 0.01
TRAIN_START = 2007
TRAIN_END = 2021
TEST_START = 2022
TEST_END = 2025

print("1. Loading and cleaning spatial events...")
z = pd.read_csv(ZONING_CSV, low_memory=False)
z["App_Date"] = pd.to_datetime(z["App_Date"], errors="coerce")
z = z.dropna(subset=["App_Date", "latitude", "longitude"])
z["year"] = z["App_Date"].dt.year
z = z[(z["year"] >= TRAIN_START) & (z["year"] <= TEST_END)].copy()

# 2. Define spatial bounding box over Austin
min_lat, max_lat = z["latitude"].min(), z["latitude"].max()
min_lon, max_lon = z["longitude"].min(), z["longitude"].max()

print(f"Bounding Box: Lat({min_lat:.3f}, {max_lat:.3f}), Lon({min_lon:.3f}, {max_lon:.3f})")

# Create grid bins
lon_bins = np.linspace(min_lon, max_lon, GRID_SIZE + 1)
lat_bins = np.linspace(min_lat, max_lat, GRID_SIZE + 1)

# Assign points to bins
z["lon_idx"] = np.digitize(z["longitude"], lon_bins) - 1
z["lat_idx"] = np.digitize(z["latitude"], lat_bins) - 1
# Clip max indices to fit within GRID_SIZE (edge cases)
z["lon_idx"] = z["lon_idx"].clip(0, GRID_SIZE - 1)
z["lat_idx"] = z["lat_idx"].clip(0, GRID_SIZE - 1)

print("2. Constructing the 3D Spatiotemporal Field (Space x Time)...")
# Aggregate counts per cell per year
counts = z.groupby(["year", "lat_idx", "lon_idx"]).size().reset_index(name="count")

# Create the full Cartesian grid for every cell and every year
years = np.arange(TRAIN_START, TEST_END + 1)
lats = np.arange(GRID_SIZE)
lons = np.arange(GRID_SIZE)
grid_idx = pd.MultiIndex.from_product([years, lats, lons], names=["year", "lat_idx", "lon_idx"])
grid = pd.DataFrame(index=grid_idx).reset_index()

# Merge actual counts onto the full grid (filling 0 for empty space)
grid = grid.merge(counts, on=["year", "lat_idx", "lon_idx"], how="left")
grid["count"] = grid["count"].fillna(0)

# Calculate normalized coordinates for neural input
grid["norm_lon"] = (grid["lon_idx"] / GRID_SIZE) * 2 - 1
grid["norm_lat"] = (grid["lat_idx"] / GRID_SIZE) * 2 - 1
grid["norm_year"] = (grid["year"] - TRAIN_START) / (TEST_END - TRAIN_START) * 2 - 1

print("2.5 Hydrating Grid Cells with Neighborhood Demographics...")
base = pd.read_csv(SPATIAL_CSV, low_memory=False)
base["parcel_id_10"] = base["standardized_tcad_id"].astype(str).str.zfill(10)

def safe_pid(x):
    try: return str(int(float(x))).zfill(10)
    except: return None
z["parcel_id_10"] = z["parcel_id_10"].map(safe_pid)

# Merge demographics onto the cases which have coordinates
cases_with_demo = z.merge(base, on="parcel_id_10", how="left")

DEMO_COLS = ["total_population", "median_household_income", "median_home_value", 
             "renter_share", "race_white", "race_hispanic", "median_age"]

# Average demographics per spatial cell (based on cases falling in that cell)
cell_demos = cases_with_demo.groupby(["lat_idx", "lon_idx"])[DEMO_COLS].mean().reset_index()

# Forward fill missing spatial cells with city-wide means to prevent NaNs
for c in DEMO_COLS:
    cell_demos[c] = cell_demos[c].fillna(cases_with_demo[c].mean())

# Normalize demographics (Z-score)
for c in DEMO_COLS:
    cell_demos[f"norm_{c}"] = (cell_demos[c] - cell_demos[c].mean()) / (cell_demos[c].std() + 1e-8)

NORM_DEMO_COLS = [f"norm_{c}" for c in DEMO_COLS]
grid = grid.merge(cell_demos[["lat_idx", "lon_idx"] + NORM_DEMO_COLS], on=["lat_idx", "lon_idx"], how="left")

# Fill any fully empty ocean/out-of-bounds cells with 0 (mean)
for c in NORM_DEMO_COLS:
    grid[c] = grid[c].fillna(0.0)

INPUT_COLS = ["norm_lon", "norm_lat", "norm_year"] + NORM_DEMO_COLS

# Train / Test split
train_mask = grid["year"] <= TRAIN_END
test_mask = grid["year"] >= TEST_START

X_train = torch.tensor(grid.loc[train_mask, INPUT_COLS].values, dtype=torch.float32)
y_train = torch.tensor(grid.loc[train_mask, "count"].values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(grid.loc[test_mask, INPUT_COLS].values, dtype=torch.float32)
y_test = torch.tensor(grid.loc[test_mask, "count"].values, dtype=torch.float32).unsqueeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"3. Building Neural Point Process MLP on device: {device} | Input dim: {len(INPUT_COLS)}")

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

class NeuralPointProcess(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Neural network outputs log(lambda)
        # We use softplus to ensure lambda is strictly positive
        return torch.nn.functional.softplus(self.net(x))

model = NeuralPointProcess(len(INPUT_COLS)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Poisson Negative Log-Likelihood: lambda - y * log(lambda)
def poisson_nll(lam, y):
    # Add epsilon to prevent log(0)
    return torch.mean(lam - y * torch.log(lam + 1e-8))

print("4. Training Continuous Spatial Intensity Field...")
best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    lam_train = model(X_train)
    loss = poisson_nll(lam_train, y_train)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            lam_test = model(X_test)
            val_loss = poisson_nll(lam_test, y_test)
            
            # Print mean lambda to ensure it hasn't collapsed
            mean_lam = lam_train.mean().item()
            print(f"[Epoch {epoch:04d}/{EPOCHS}] Train NLL: {loss.item():.4f} | Val NLL: {val_loss.item():.4f} | Avg $\lambda$: {mean_lam:.4f}")

print("\n5. Generating Visualization for 2023 Out-Of-Time Forecast...")
# We will predict the continuous spatial surface for the year 2023
model.eval()
target_year = 2023
target_norm_year = (target_year - TRAIN_START) / (TEST_END - TRAIN_START) * 2 - 1

# Generate a high-res querying grid
RES = 100
query_lon = np.linspace(-1, 1, RES)
query_lat = np.linspace(-1, 1, RES)
lon_mesh, lat_mesh = np.meshgrid(query_lon, query_lat)
year_mesh = np.full_like(lon_mesh, target_norm_year)

# We must also attach the spatial demographics to the query grid!
# We can map the query lat/lon back to the 50x50 grid cells to borrow their demographics
query_lon_idx = ((lon_mesh + 1) / 2 * GRID_SIZE).astype(int).clip(0, GRID_SIZE - 1)
query_lat_idx = ((lat_mesh + 1) / 2 * GRID_SIZE).astype(int).clip(0, GRID_SIZE - 1)

query_df = pd.DataFrame({
    "lat_idx": query_lat_idx.flatten(),
    "lon_idx": query_lon_idx.flatten(),
    "norm_lon": lon_mesh.flatten(),
    "norm_lat": lat_mesh.flatten(),
    "norm_year": year_mesh.flatten()
})

query_df = query_df.merge(cell_demos[["lat_idx", "lon_idx"] + NORM_DEMO_COLS], on=["lat_idx", "lon_idx"], how="left")
for c in NORM_DEMO_COLS:
    query_df[c] = query_df[c].fillna(0.0)

query_tensor = torch.tensor(query_df[INPUT_COLS].values, dtype=torch.float32).to(device)

with torch.no_grad():
    pred_lambda = model(query_tensor).cpu().numpy().reshape(RES, RES)

# Get actual cases in 2023
actual_2023 = z[z["year"] == target_year]

plt.figure(figsize=(10, 8), facecolor="#121212")
ax = plt.gca()
ax.set_facecolor("#121212")

# Plot the neural intensity heatmap
# Note: imshow origin is top by default, so we use origin='lower' to match lat/lon
im = plt.imshow(pred_lambda, extent=[min_lon, max_lon, min_lat, max_lat], 
           origin='lower', cmap="magma", alpha=0.8)

# Overlay actual case coordinates
plt.scatter(actual_2023["longitude"], actual_2023["latitude"], 
            color="cyan", s=15, edgecolors="white", linewidths=0.5, label="Actual 2023 Zoning Cases")

plt.colorbar(im, label="Predicted Filing Intensity ($\lambda$)")
plt.title("Neural Poisson Point Process: 2023 Spatial Hazard Topology", color="white", fontsize=14)
plt.xlabel("Longitude", color="white")
plt.ylabel("Latitude", color="white")
ax.tick_params(colors="white")
plt.legend(facecolor="#121212", labelcolor="white", edgecolor="#444444")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight", facecolor="#121212")
print(f"SUCCESS: Visualization saved to {OUT_PLOT}")
