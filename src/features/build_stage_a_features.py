import pandas as pd

from src.data_io.schema import REGISTRY_DIR, ROOT_DIR, ensure_dirs, save_registry


def build_stage_a_features() -> pd.DataFrame:
    """Build a small, deterministic selection-correction feature view."""

    ensure_dirs()
    universe_path = REGISTRY_DIR / "case_universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(f"Case universe not found: {universe_path}")

    universe = pd.read_parquet(universe_path).copy()
    X = universe[["case_id", "filing_date"]].copy()
    X["as_of_date"] = X["filing_date"]
    X["feature_view"] = "stage_a_hazard"
    filing_dates = pd.to_datetime(X["filing_date"], errors="coerce")  # pyright: ignore[reportUnknownMemberType]
    X["filing_year"] = pd.Series(filing_dates, index=X.index).dt.year.astype("Float64")
    X["year_centered"] = X["filing_year"] - X["filing_year"].median()
    X["district_missing"] = universe["council_district"].isna().astype(int) if "council_district" in universe.columns else 0
    X["selection_proxy"] = X["year_centered"].abs().fillna(0).gt(2).astype(int)

    interim_path = ROOT_DIR / "data" / "interim" / "stage_a_features_raw.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(interim_path, index=False)

    save_registry(X[["case_id", "as_of_date", "feature_view"]].drop_duplicates(), "feature_registry")
    return X

if __name__ == "__main__":
    build_stage_a_features()
