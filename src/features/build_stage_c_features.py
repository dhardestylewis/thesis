"""Build the canonical filing-date feature view for Stage C."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd

from src.data_io.schema import REGISTRY_DIR, ROOT_DIR, WAREHOUSE_DIR, ensure_dirs, save_registry

EXCLUDE_COLUMNS = {
    "case_id",
    "case_number",
    "filing_date",
    "date_filed",
    "submitted_date",
    "year",
    "filing_year",
    "is_protested",
    "protested",
    "petition_crossed",
    "threshold_crossed",
    "label_version",
    "reconstructed_petition_share",
    "clerk_validity_observed",
    "procedural_defect_signal",
}

to_datetime = cast(Callable[..., Any], getattr(pd, "to_datetime"))
to_numeric = cast(Callable[..., Any], getattr(pd, "to_numeric"))


def _load_source_frame(source_path: Optional[str] = None) -> pd.DataFrame:
    path = Path(source_path) if source_path else WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv"
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def build_stage_c_features(
    source_path: Optional[str] = None,
    case_universe_path: Optional[str] = None,
    output_path: Optional[str] = None,
    feature_view_name: str = "filing_date_public",
) -> pd.DataFrame:
    """Export the canonical Stage C matrix and its feature registry."""

    ensure_dirs()
    source = _load_source_frame(source_path)
    case_universe = None
    if case_universe_path:
        case_universe = pd.read_parquet(Path(case_universe_path))
    else:
        default_universe = REGISTRY_DIR / "case_universe.parquet"
        if default_universe.exists():
            case_universe = pd.read_parquet(default_universe)
            case_universe = case_universe.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)

    case_col = next((c for c in ["case_id", "case_number"] if c in source.columns), None)
    if case_col is None:
        raise ValueError("Could not identify a case identifier column for Stage C features.")

    if "filing_date" in source.columns:
        as_of = to_datetime(source["filing_date"], errors="coerce")
    elif "date_filed" in source.columns:
        as_of = to_datetime(source["date_filed"], errors="coerce")
    elif "year" in source.columns:
        as_of = to_datetime(source["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    else:
        as_of = pd.Series(pd.NaT, index=source.index)

    numeric = source.select_dtypes(include=[np.number]).copy()
    numeric = numeric[[c for c in numeric.columns if c not in EXCLUDE_COLUMNS]].copy()
    feature_matrix = pd.concat(
        [
            pd.DataFrame(
                {
                    "case_id": source[case_col].astype(str),
                    "as_of_date": as_of.dt.strftime("%Y-%m-%d") if hasattr(as_of, "dt") else pd.NA,
                    "feature_view": feature_view_name,
                },
                index=source.index,
            ),
            numeric.reset_index(drop=True),
        ],
        axis=1,
    )

    # Add a small number of deterministic derived features if they can help the model.
    if "year" in source.columns and "filing_year" not in feature_matrix.columns:
        feature_matrix["filing_year"] = to_numeric(source["year"], errors="coerce")

    if case_universe is not None and "council_district" in case_universe.columns:
        district_map = case_universe[["case_id", "council_district"]].copy()
        district_map["council_district"] = to_numeric(district_map["council_district"], errors="coerce")
        district_map = district_map.drop_duplicates(subset=["case_id"])
        feature_matrix = feature_matrix.merge(district_map, on="case_id", how="left")
        
    # INJECT EXACT GEOMETRIC PETITION INTENSITY
    geom_pet_path = Path(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\exact_geometric_petition_intensity.csv")
    if geom_pet_path.exists():
        pet_df = pd.read_csv(geom_pet_path)
        pet_df["case_number"] = pet_df["case_number"].str.strip()
        
        # We need a crosswalk from case_id to case_number if feature_matrix uses case_id
        if "case_id" in feature_matrix.columns and "case_number" not in feature_matrix.columns:
            if "case_number" in source.columns:
                cw = source[["case_id", "case_number"]].drop_duplicates()
                cw["case_number"] = cw["case_number"].str.strip()
                pet_df = pet_df.merge(cw, on="case_number", how="inner")
                pet_df = pet_df.drop(columns=["case_number"])
                
        # Merge it
        feature_matrix = feature_matrix.merge(pet_df, on="case_id" if "case_id" in pet_df.columns else "case_number", how="left")
        if "label_exact_geometric_petition_pct" in feature_matrix.columns:
            feature_matrix["label_exact_geometric_petition_pct"].fillna(0, inplace=True)

    feature_matrix = feature_matrix.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)

    feature_matrix = feature_matrix.sort_values(["case_id", "as_of_date"]).reset_index(drop=True)

    matrix_path = Path(output_path) if output_path else ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_parquet(matrix_path, index=False)

    registry = feature_matrix[["case_id", "as_of_date", "feature_view"]].copy()
    save_registry(registry.drop_duplicates(), "feature_registry")
    return feature_matrix


if __name__ == "__main__":
    build_stage_c_features()
