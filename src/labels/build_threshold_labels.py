"""Build the frozen case universe and threshold-crossing label registry.

The thesis now treats label construction as an auditable layer preceding model
training. This module therefore produces three explicit label versions:

- label_v1_reconstructed_threshold_crossing
- label_v2_strict_reconstructed_threshold_crossing
- label_v3_hand_validated_subsample (or an audited stub when no file is present)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, cast

import numpy as np
import pandas as pd

from src.data_io.schema import REGISTRY_DIR, WAREHOUSE_DIR, WAREHOUSE_MASTER, ensure_dirs

CASE_ID_CANDIDATES = ["case_id", "case_number", "case_no", "zoning_case_id"]
YEAR_CANDIDATES = ["year", "filing_year"]
DISTRICT_CANDIDATES = ["council_district", "ldb_council_district", "council_district_x"]
PROTEST_CANDIDATES = ["is_protested", "protested", "petition_crossed", "threshold_crossed"]
DATE_CANDIDATES = ["filing_date", "date_filed", "submitted_date"]

to_datetime_series = cast(Callable[..., pd.Series], getattr(pd, "to_datetime"))
to_numeric_series = cast(Callable[..., pd.Series], getattr(pd, "to_numeric"))


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _load_source_frame(source_path: Optional[str] = None) -> pd.DataFrame:
    path = Path(source_path) if source_path else WAREHOUSE_MASTER
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def build_case_universe(
    source_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Freeze the analytic case universe used by every downstream task."""

    ensure_dirs()
    df = _load_source_frame(source_path)

    case_col = _first_present(df.columns, CASE_ID_CANDIDATES)
    if case_col is None:
        raise ValueError("Could not identify a case identifier column.")

    year_col = _first_present(df.columns, YEAR_CANDIDATES)
    date_col = _first_present(df.columns, DATE_CANDIDATES)
    district_col = _first_present(df.columns, DISTRICT_CANDIDATES)

    universe = pd.DataFrame({"case_id": df[case_col].astype(str)})
    if year_col is not None:
        universe["filing_year"] = to_numeric_series(df[year_col], errors="coerce").astype("Int64")
    elif date_col is not None:
        filing_dates = to_datetime_series(df[date_col], errors="coerce")
        universe["filing_year"] = filing_dates.dt.year.astype("Int64")
    else:
        universe["filing_year"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if date_col is not None:
        universe["filing_date"] = to_datetime_series(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    elif year_col is not None:
        numeric_year = cast(Any, to_numeric_series(df[year_col], errors="coerce"))
        universe["filing_date"] = numeric_year.fillna(0).astype(int).astype(str) + "-01-01"
    else:
        universe["filing_date"] = pd.NA

    if district_col is not None:
        universe["council_district"] = df[district_col]
    else:
        universe["council_district"] = pd.NA

    discretionary_col = next((c for c in ["is_discretionary", "discretionary", "discretionary_case"] if c in df.columns), None)
    if discretionary_col is not None:
        universe["is_discretionary"] = df[discretionary_col].astype(bool)
    else:
        universe["is_discretionary"] = True

    universe = universe[universe["is_discretionary"]].copy()
    universe = universe.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)

    universe["sample_inclusion_reason"] = np.where(
        universe["is_discretionary"],
        "discretionary_case",
        "non_discretionary_case",
    )
    universe["geometry_available"] = True
    universe["transcript_available"] = "transcript" in {c.lower() for c in df.columns}
    universe["label_quality_flag"] = "unreviewed"

    output = Path(output_path) if output_path else REGISTRY_DIR / "case_universe.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(output, index=False)
    return universe


def _build_primary_label_frame(universe: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    case_col = _first_present(df.columns, CASE_ID_CANDIDATES)
    protest_col = _first_present(df.columns, PROTEST_CANDIDATES)

    labels = universe[["case_id"]].copy()
    if case_col is not None:
        df = df.drop_duplicates(subset=[case_col], keep="first").copy()
    labels = labels.merge(
        df[[case_col, protest_col]].copy() if case_col and protest_col else pd.DataFrame({"case_id": labels["case_id"]}),
        left_on="case_id",
        right_on=case_col if case_col else "case_id",
        how="left",
    )
    if case_col and case_col in labels.columns:
        labels = labels.drop(columns=[case_col])

    raw_signal = to_numeric_series(labels[protest_col], errors="coerce") if protest_col else pd.Series(np.nan, index=labels.index)
    reconstructed = cast(Any, raw_signal).fillna(0.0).clip(lower=0.0, upper=1.0)

    case_level = pd.DataFrame(
        {
            "case_id": labels["case_id"].astype(str),
            "reconstructed_petition_share": reconstructed.astype(float),
            "signal_observed": raw_signal.notna(),
        }
    )
    case_level = (
        case_level.groupby("case_id", as_index=False)
        .agg(
            reconstructed_petition_share=("reconstructed_petition_share", "max"),
            signal_observed=("signal_observed", "max"),
            source_file_count=("signal_observed", "sum"),
        )
    )

    out = universe[["case_id"]].copy()
    out = out.merge(case_level, on="case_id", how="left")
    out["reconstructed_petition_share"] = cast(Any, out["reconstructed_petition_share"]).fillna(0.0).astype(float)
    out["signal_observed"] = cast(Any, out["signal_observed"]).fillna(False).astype(bool)
    out["source_file_count"] = cast(Any, out["source_file_count"]).fillna(0).astype(int)
    out["threshold_crossed"] = (out["reconstructed_petition_share"] >= 0.20).astype(int)
    out["clerk_validity_observed"] = pd.NA
    out["procedural_defect_signal"] = pd.NA
    out["label_confidence"] = np.where(out["signal_observed"], 0.85, 0.45)
    out["source_provenance"] = np.where(out["signal_observed"], "warehouse_proxy", "missing_proxy")
    out = out.drop(columns=["signal_observed"])
    out["label_notes"] = "reconstructed threshold-crossing proxy"
    return out


def build_threshold_labels(
    case_universe_path: Optional[str] = None,
    petition_records_path: Optional[str] = None,
    hand_validated_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build the label registry with three explicit label versions."""

    ensure_dirs()
    universe_path = Path(case_universe_path) if case_universe_path else REGISTRY_DIR / "case_universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(f"Case universe not found: {universe_path}")

    universe = pd.read_parquet(universe_path)
    source_df = _load_source_frame(petition_records_path)
    primary = _build_primary_label_frame(universe, source_df)

    v1 = primary.copy()
    v1["label_version"] = "label_v1_reconstructed_threshold_crossing"

    v2 = primary.copy()
    v2["label_version"] = "label_v2_strict_reconstructed_threshold_crossing"
    v2["threshold_crossed"] = (v2["reconstructed_petition_share"] >= 0.25).astype(int)
    v2["label_confidence"] = v2["label_confidence"].clip(upper=0.80)
    v2["label_notes"] = "strict threshold sensitivity"

    if hand_validated_path:
        audited_path = Path(hand_validated_path)
    else:
        audited_path = REGISTRY_DIR / "label_v3_hand_validated_subsample.parquet"

    if audited_path.exists():
        v3 = pd.read_parquet(audited_path).copy()
        if "case_id" not in v3.columns:
            raise ValueError("The audited label file must contain a case_id column.")
        if "threshold_crossed" not in v3.columns:
            raise ValueError("The audited label file must contain threshold_crossed.")
        if "label_version" not in v3.columns:
            v3["label_version"] = "label_v3_hand_validated_subsample"
        v3 = v3[[c for c in v3.columns if c in {
            "case_id", "label_version", "reconstructed_petition_share", "threshold_crossed",
            "clerk_validity_observed", "procedural_defect_signal", "label_confidence",
            "source_file_count", "source_provenance", "label_notes"
        }]].copy()
    else:
        sample_n = min(200, len(v1))
        v3 = v1.sample(n=sample_n, random_state=42).copy() if sample_n else v1.head(0).copy()
        v3["label_version"] = "label_v3_hand_validated_subsample"
        v3["source_provenance"] = "audit_stub"
        v3["label_confidence"] = 1.0
        v3["label_notes"] = "stubbed audited subset; replace with hand validation file"

    label_registry = pd.concat([v1, v2, v3], ignore_index=True, sort=False)
    label_registry = label_registry.sort_values(["label_version", "case_id"]).reset_index(drop=True)

    output = Path(output_path) if output_path else REGISTRY_DIR / "label_registry.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    label_registry.to_parquet(output, index=False)

    audit_rows: list[dict[str, Any]] = []
    audit = v1.merge(v3[["case_id", "threshold_crossed"]], on="case_id", how="inner", suffixes=("_v1", "_v3"))
    if not audit.empty:
        agreement = float((audit["threshold_crossed_v1"] == audit["threshold_crossed_v3"]).mean())
        audit_rows.append({
            "metric": "agreement",
            "value": agreement,
            "sample_size": int(len(audit)),
            "audit_mode": "hand_validated" if audited_path.exists() and hand_validated_path else "stub",
        })
        audit_rows.append({
            "metric": "disagreement_rate",
            "value": float(1.0 - agreement),
            "sample_size": int(len(audit)),
            "audit_mode": "hand_validated" if audited_path.exists() and hand_validated_path else "stub",
        })
    else:
        audit_rows.append({"metric": "agreement", "value": None, "sample_size": 0, "audit_mode": "empty"})

    audit_path = REGISTRY_DIR / "label_audit_results.json"
    audit_path.write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    return label_registry


if __name__ == "__main__":
    build_case_universe()
    build_threshold_labels()
