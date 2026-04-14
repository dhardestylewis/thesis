"""Shared paths and registry helpers for the thesis pipeline.

The pipeline is built around stable, auditable registries. This module keeps
path handling and run-key validation in one place so the rest of the code can
stay focused on the scientific task instead of filesystem plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
REGISTRY_DIR = ROOT_DIR / "registries"
# Canonical Stage C prediction horizon key written by train_stage_c (dedupes legacy aliases in registries).
PRIMARY_STAGE_C_HORIZON = "filing"
CONFIG_DIR = ROOT_DIR / "configs"
SRC_DIR = ROOT_DIR / "src"
WAREHOUSE_DIR = ROOT_DIR / "Data" / "Warehouse_As_Of"


class RunKey(TypedDict, total=False):
    case_id: str
    as_of_date: str
    horizon: str
    label_version: str
    feature_view: str
    split_id: str
    model_family: str
    seed: Optional[int]
    calibration_method: Optional[str]


def ensure_dirs() -> None:
    for directory in [DATA_DIR, REGISTRY_DIR, CONFIG_DIR, SRC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for subdir in ["raw", "interim", "final"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def load_registry(name: str) -> pd.DataFrame:
    path = REGISTRY_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def save_registry(df: pd.DataFrame, name: str) -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = REGISTRY_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def registry_path(name: str) -> Path:
    return REGISTRY_DIR / f"{name}.parquet"


def validate_run_key_columns(df: pd.DataFrame) -> None:
    required = ["case_id", "as_of_date", "horizon", "label_version", "feature_view", "split_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing run-key columns: {missing}")
