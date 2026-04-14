"""Build the manuscript-safe metrics manifest from registered analysis outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from src.data_io.schema import PRIMARY_STAGE_C_HORIZON, REGISTRY_DIR, ROOT_DIR, ensure_dirs

PRIMARY_TASK_ID = "STAGE_C_FILING_MAIN"
PRIMARY_SPLIT_ID = "TEMP_OOD_2023_MAIN"
PRIMARY_MODEL = "CatBoost"
PRIMARY_HORIZON = "filing"
PRIMARY_CALIBRATION = "identity_noop"


def _repo_root() -> Path:
    return ROOT_DIR


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _record(metric_id: str, value: Any, source_artifact: str, *, ci_low: Optional[float] = None, ci_high: Optional[float] = None, task_id: str = PRIMARY_TASK_ID, split_id: str = PRIMARY_SPLIT_ID, model_family: str = PRIMARY_MODEL, horizon: str = PRIMARY_HORIZON, calibration_state: str = PRIMARY_CALIBRATION, seed_policy: str = "single_seed") -> Dict[str, Any]:
    value_as_float = _to_float_or_none(value)
    ci_low_value = _to_float_or_none(ci_low)
    ci_high_value = _to_float_or_none(ci_high)
    if isinstance(value, (dict, list, str, bool)):
        normalized_value: Any = cast(Any, value)
    else:
        normalized_value = value_as_float
    return {
        "metric_id": metric_id,
        "value": normalized_value,
        "ci_low": ci_low_value,
        "ci_high": ci_high_value,
        "task_id": task_id,
        "split_id": split_id,
        "model_family": model_family,
        "horizon": horizon,
        "calibration_state": calibration_state,
        "seed_policy": seed_policy,
        "source_artifact": source_artifact,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _from_evaluation(eval_data: Dict[str, Any], source_artifact: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ranking = eval_data.get("ranking", {})
    calibration = eval_data.get("calibration", {})
    thresholded = eval_data.get("thresholded", {})

    if ranking:
        rows.append(_record("metricHeadlinePRAUC", ranking.get("pr_auc"), source_artifact))
        rows.append(_record("metricBootstrapFiling", ranking.get("pr_auc"), source_artifact, ci_low=ranking.get("pr_auc_ci_low"), ci_high=ranking.get("pr_auc_ci_high")))
        rows.append(_record("metricTopDecilePrecision", ranking.get("top_decile_precision"), source_artifact))
        rows.append(_record("metricTopDecileLift", ranking.get("top_decile_lift"), source_artifact))
    if calibration:
        ece_val = calibration.get("ece")
        rows.append(_record("metricHeadlineECE", ece_val, source_artifact))
        rows.append(_record("metricECE", ece_val, source_artifact))
        rows.append(_record("metricECEOODBootstrapCalibrated", ece_val, source_artifact))
        rows.append(_record("metricACE", calibration.get("ace"), source_artifact))
        rows.append(_record("metricBrierScore", calibration.get("brier"), source_artifact))
    if thresholded:
        for key, metric_id in [
            ("precision_at_0_30", "metricPrecisionAtThirty"),
            ("recall_at_0_30", "metricRecallAtThirty"),
            ("precision_at_0_50", "metricPrecisionAtFifty"),
            ("recall_at_0_50", "metricRecallAtFifty"),
        ]:
            if key in thresholded:
                rows.append(_record(metric_id, thresholded[key], source_artifact))
    return rows


def build_metrics_manifest(evaluation_outputs_dir: Optional[str] = None, output_json: Optional[str] = None) -> List[Dict[str, Any]]:
    """Assemble the final thesis metrics manifest from registered outputs."""

    ensure_dirs()
    root = _repo_root()
    outputs_dir = Path(evaluation_outputs_dir) if evaluation_outputs_dir else REGISTRY_DIR
    out_path = Path(output_json) if output_json else root / "reporting" / "final_metrics_manifest.json"

    universe = _safe_read_parquet(REGISTRY_DIR / "case_universe.parquet")
    labels = _safe_read_parquet(REGISTRY_DIR / "label_registry.parquet")
    preds = _safe_read_parquet(REGISTRY_DIR / "prediction_registry.parquet")
    ablation = _safe_read_parquet(REGISTRY_DIR / "ablation_results.parquet")

    eval_json = cast(Dict[str, Any], _safe_read_json(outputs_dir / "evaluation_results.json"))
    audit_json = _safe_read_json(REGISTRY_DIR / "label_audit_results.json")
    meta_json = _safe_read_json(REGISTRY_DIR / "meta_attribution_object.json")

    records: List[Dict[str, Any]] = []

    if not universe.empty:
        records.append(_record("metricBaselineCases", len(universe), "registries/case_universe.parquet", task_id=PRIMARY_TASK_ID, split_id=PRIMARY_SPLIT_ID))
        if "council_district" in universe.columns:
            records.append(_record("metricNDistricts", universe["council_district"].dropna().nunique(), "registries/case_universe.parquet", task_id=PRIMARY_TASK_ID, split_id=PRIMARY_SPLIT_ID))

    label_slice = labels[labels["label_version"] == "label_v1_reconstructed_threshold_crossing"] if not labels.empty and "label_version" in labels.columns else pd.DataFrame()
    if not label_slice.empty:
        records.append(_record("metricBaseRate", label_slice["threshold_crossed"].mean(), "registries/label_registry.parquet"))
        records.append(_record("metricBaseRateN", label_slice["threshold_crossed"].sum(), "registries/label_registry.parquet"))

    if not preds.empty:
        main_preds = preds[(preds["model_family"] == PRIMARY_MODEL) & (preds["split_id"] == PRIMARY_SPLIT_ID)]
        if "horizon" in main_preds.columns:
            main_preds = main_preds.loc[main_preds["horizon"] == PRIMARY_STAGE_C_HORIZON]
        if not main_preds.empty:
            records.append(_record("metricNTestTotal", int(len(main_preds)), "registries/prediction_registry.parquet"))
            records.append(_record("metricNTestPositive", int(main_preds["y_true"].sum()), "registries/prediction_registry.parquet"))
            records.append(_record("metricMeanProbTP", main_preds.loc[main_preds["y_true"] == 1, "y_score_raw"].mean(), "registries/prediction_registry.parquet"))
            records.append(_record("metricMeanProbTN", main_preds.loc[main_preds["y_true"] == 0, "y_score_raw"].mean(), "registries/prediction_registry.parquet"))
            top_decile_cut = main_preds["y_score_raw"].quantile(0.9)
            top_decile = main_preds[main_preds["y_score_raw"] >= top_decile_cut]
            if not top_decile.empty:
                records.append(_record("metricTopDecileTP", top_decile["y_true"].sum(), "registries/prediction_registry.parquet"))
                records.append(_record("metricTopDecileN", len(top_decile), "registries/prediction_registry.parquet"))

    records.extend(_from_evaluation(eval_json, "registries/evaluation_results.json"))

    if audit_json:
        if isinstance(audit_json, list):
            audit_list = cast(List[Any], audit_json)
            first: Any = audit_list[0] if audit_list else {}
        else:
            first = audit_json
        if isinstance(first, dict):
            first_dict = cast(Dict[str, Any], first)
            label_value = first_dict.get("value", first_dict.get("agreement"))
            if label_value is not None:
                records.append(_record("metricLabelFidelity", label_value, "registries/label_audit_results.json"))
            label_n = first_dict.get("sample_size")
            if label_n is not None:
                records.append(_record("metricLabelAuditN", label_n, "registries/label_audit_results.json"))

    if not ablation.empty and {"cluster", "delta_prauc"}.issubset(ablation.columns):
        best = ablation.sort_values("delta_prauc", ascending=False).iloc[0]
        records.append(_record("metricLargestClusterDelta", best["delta_prauc"], "registries/ablation_results.parquet", task_id="STAGE_C_INTERPRETATION_MAIN", split_id=PRIMARY_SPLIT_ID, model_family=str(best.get("model_family", PRIMARY_MODEL))))

    if meta_json:
        if isinstance(meta_json, list):
            meta_list = cast(List[Any], meta_json)
            meta_rows = [cast(Dict[str, Any], row) for row in meta_list if isinstance(row, dict)]
            consensus_count = sum(1 for row in meta_rows if row.get("is_consensus"))
            unstable_count = sum(
                1
                for row in meta_rows
                if row.get("std") is not None
                and row.get("count", 0) > 1
                and (row.get("std", 0) or 0) > 0
            )
        else:
            meta_dict = cast(Dict[str, Any], meta_json)
            consensus_count = len(cast(List[Any], meta_dict.get("cross_model_consensus_features", [])))
            unstable_count = len(cast(List[Any], meta_dict.get("unstable_features", [])))
        records.append(_record("metricConsensusClusterCount", consensus_count, "registries/meta_attribution_object.json", task_id="STAGE_C_INTERPRETATION_MAIN", split_id=PRIMARY_SPLIT_ID))
        records.append(_record("metricUnstableClusterCount", unstable_count, "registries/meta_attribution_object.json", task_id="STAGE_C_INTERPRETATION_MAIN", split_id=PRIMARY_SPLIT_ID))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    build_metrics_manifest()
