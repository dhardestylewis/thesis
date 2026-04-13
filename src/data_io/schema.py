import os
from pathlib import Path

# Paths.py
ROOT_DIR = Path(r"c:\Users\dhl\data\thesis\thesis")
DATA_DIR = ROOT_DIR / "data"
REGISTRY_DIR = ROOT_DIR / "registries"
CONFIG_DIR = ROOT_DIR / "configs"
SRC_DIR = ROOT_DIR / "src"

# Warehouse source
WAREHOUSE_DIR = ROOT_DIR / "Data" / "Warehouse_As_Of"

def ensure_dirs():
    for d in [DATA_DIR, REGISTRY_DIR, CONFIG_DIR, SRC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(exist_ok=True)
    (DATA_DIR / "interim").mkdir(exist_ok=True)
    (DATA_DIR / "final").mkdir(exist_ok=True)

# Schema.py (TypedDict or just documentation of the contract)
from typing import TypedDict, Optional

class RunKey(TypedDict):
    case_id: str
    as_of_date: str
    horizon: str
    label_version: str
    feature_view: str
    split_id: str
    model_family: str
    seed: Optional[int]
    calibration_method: Optional[str]

# Registry helper
import pandas as pd

def load_registry(name: str) -> pd.DataFrame:
    path = REGISTRY_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

def save_registry(df: pd.DataFrame, name: str):
    df.to_parquet(REGISTRY_DIR / f"{name}.parquet", index=False)
