"""
precompute_hypergrid.py  —  Calculates marginal causal effects for all 120 height tiers
across all 270k parcels. Saves to a 390MB binary cache for instant API serving.
"""
import numpy as np
import joblib
import os
import time
from pathlib import Path

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
MODELS_PATH = ROOT / "Data/Zoning_Cases/causal_models.pkl"
X_BASE_PATH = ROOT / "Data/Zoning_Cases/X_base.npy"
CACHE_OUT   = ROOT / "Data/Zoning_Cases/inference_cache.npy"

print(f"Loading Models and Base Data...", flush=True)
models = joblib.load(MODELS_PATH)
cf_joint = models['cf_joint']
cf_withd = models['cf_withd']
X_base = np.load(X_BASE_PATH)

n_parcels = len(X_base)
heights = np.arange(5, 121, 1) # 116 steps
n_heights = len(heights)

# Storage: [heights, parcels, 3 metrics]
# metrics: [delay_marginal, attrition_marginal, withdrawal_marginal]
hypergrid = np.zeros((n_heights, n_parcels, 3), dtype=np.float32)

print(f"Starting Precomputation for {n_heights} height tiers...", flush=True)
t_start = time.time()

for i, h in enumerate(heights):
    t0 = time.time()
    # Update height feature in X_base
    X_base[:, 0] = h
    
    # Calculate marginal effects (slope of the dose response)
    # T0=0, T1=1 gives the effect of moving from 0% to 100% petition dose
    m_joint = cf_joint.effect(X_base, T0=0.0, T1=1.0)
    m_withd = cf_withd.effect(X_base, T0=0.0, T1=1.0)
    
    hypergrid[i, :, 0] = m_joint[:, 1].astype(np.float32) # delay_marginal
    hypergrid[i, :, 1] = m_joint[:, 0].astype(np.float32) # attrition_marginal
    hypergrid[i, :, 2] = m_withd.astype(np.float32)      # withdrawal_marginal
    
    print(f"  [{i+1}/{n_heights}] Height {h}ft done in {time.time()-t0:.2f}s", flush=True)

print(f"\nTotal precomputation took {(time.time()-t_start)/60:.1f} minutes.")
print(f"Saving Hypergrid to {CACHE_OUT} ({hypergrid.nbytes/1e6:.1f} MB)...", flush=True)
np.save(CACHE_OUT, hypergrid)
print("Done!")
