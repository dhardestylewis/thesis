"""Build immutable split registries for the thesis experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional, cast

import numpy as np
import pandas as pd

from src.data_io.schema import REGISTRY_DIR, ensure_dirs

PRIMARY_SPLIT_ID = "TEMP_OOD_2023_MAIN"
CV_SPLIT_ID = "CV5_IN_DIST_MAIN"
ROLLING_SPLIT_ID = "TEMP_ROLL_PRE2019_TO_2024"
SPATIAL_SPLIT_ID = "SPATIAL_CD_HOLDOUT_D4"

to_datetime_series = cast(Callable[..., pd.Series], getattr(pd, "to_datetime"))
to_numeric_series = cast(Callable[..., pd.Series], getattr(pd, "to_numeric"))


def _stable_bucket(value: object, modulo: int) -> int:
    text = "" if pd.isna(value) else str(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def build_split_registry(case_universe_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    ensure_dirs()
    universe_path = Path(case_universe_path) if case_universe_path else REGISTRY_DIR / "case_universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(f"Case universe not found: {universe_path}")

    universe = pd.read_parquet(universe_path).copy()
    universe = universe.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)
    if "filing_year" not in universe.columns:
        if "filing_date" in universe.columns:
            universe["filing_year"] = to_datetime_series(universe["filing_date"], errors="coerce").dt.year
        else:
            universe["filing_year"] = pd.NA
    filing_year_numeric = to_numeric_series(universe["filing_year"], errors="coerce")

    rows: list[pd.DataFrame] = []

    # Primary temporal OOD split: pre-2023 train, 2023+ test.
    primary = universe[["case_id", "filing_year"]].copy()
    primary["split_id"] = PRIMARY_SPLIT_ID
    primary["split_family"] = "temporal_ood"
    primary["fold"] = 0
    primary_year = cast(object, filing_year_numeric).reindex(primary.index)
    primary["role"] = np.where(cast(object, primary_year).fillna(9999).astype(int) < 2023, "train", "test")
    primary["anchor_year"] = 2023
    primary["eval_year"] = primary["filing_year"]
    rows.append(primary)

    # In-distribution CV assignment. This registry captures fold membership;
    # downstream CV runners can derive train/valid from the fold column.
    cv = universe[["case_id", "filing_year"]].copy()
    cv["split_id"] = CV_SPLIT_ID
    cv["split_family"] = "cv5_in_distribution"
    cv["fold"] = cv["case_id"].astype(str).map(lambda x: _stable_bucket(x, 5))
    cv["role"] = "member"
    cv["anchor_year"] = cv["filing_year"]
    cv["eval_year"] = cv["filing_year"]
    rows.append(cv)

    # Rolling origin registry: same universe, but the evaluation anchor is made explicit.
    rolling = universe[["case_id", "filing_year"]].copy()
    rolling["split_id"] = ROLLING_SPLIT_ID
    rolling["split_family"] = "rolling_origin"
    rolling_year = cast(object, filing_year_numeric).reindex(rolling.index)
    rolling["fold"] = cast(object, rolling_year).fillna(0).astype(int).clip(lower=0)
    rolling["role"] = np.where(cast(object, rolling_year).fillna(0).astype(int) < 2020, "train", "test")
    rolling["anchor_year"] = 2020
    rolling["eval_year"] = rolling["filing_year"]
    rows.append(rolling)

    # Optional spatial holdout if a district field exists.
    district_col = next((c for c in ["council_district", "district"] if c in universe.columns), None)
    if district_col is not None:
        spatial = universe[["case_id", district_col, "filing_year"]].copy()
        spatial["split_id"] = SPATIAL_SPLIT_ID
        spatial["split_family"] = "spatial_holdout"
        spatial["fold"] = 4
        spatial["role"] = np.where(spatial[district_col].astype(str) == "4", "test", "train")
        spatial["anchor_year"] = spatial["filing_year"]
        spatial["eval_year"] = spatial["filing_year"]
        rows.append(spatial.drop(columns=[district_col]))

    split_registry = pd.concat(rows, ignore_index=True, sort=False)
    split_registry["case_id"] = split_registry["case_id"].astype(str)
    split_registry = split_registry.sort_values(["split_id", "case_id"]).reset_index(drop=True)

    out_path = Path(output_path) if output_path else REGISTRY_DIR / "split_registry.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_registry.to_parquet(out_path, index=False)
    return split_registry


if __name__ == "__main__":
    build_split_registry()
