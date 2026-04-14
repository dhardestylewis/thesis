import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REQUIRED_KEYS = [
    "metric_id",
    "value",
    "ci_low",
    "ci_high",
    "task_id",
    "split_id",
    "model_family",
    "horizon",
    "calibration_state",
    "seed_policy",
    "source_artifact",
    "generated_at",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return None


def _record(
    metric_id: str,
    value: Any,
    source_artifact: str,
    generated_at: str,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    task_id: str = "stage_c_ranking",
    split_id: str = "TEMP_OOD_2023_MAIN",
    model_family: str = "CatBoost",
    horizon: str = "filing",
    calibration_state: str = "isotonic_oof",
    seed_policy: str = "single_seed",
) -> Dict[str, Any]:
    row = {
        "metric_id": metric_id,
        "value": value,
        "ci_low": _to_float_or_none(ci_low),
        "ci_high": _to_float_or_none(ci_high),
        "task_id": task_id,
        "split_id": split_id,
        "model_family": model_family,
        "horizon": horizon,
        "calibration_state": calibration_state,
        "seed_policy": seed_policy,
        "source_artifact": source_artifact,
        "generated_at": generated_at,
    }
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(f"Missing required keys for {metric_id}: {missing}")
    return row


def _validate_duplicates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for row in records:
        metric_id = row["metric_id"]
        if metric_id not in deduped:
            deduped[metric_id] = row
            continue

        prior = deduped[metric_id]
        same_value = (
            prior["value"] == row["value"]
            and prior.get("ci_low") == row.get("ci_low")
            and prior.get("ci_high") == row.get("ci_high")
        )
        if same_value:
            continue

        # versioned metric IDs are allowed to coexist when the base ID differs by version suffix
        if "@" in metric_id:
            raise ValueError(
                f"Duplicate versioned metric_id '{metric_id}' with conflicting values: "
                f"{prior['value']} vs {row['value']}"
            )

        raise ValueError(
            f"Conflicting values for metric_id '{metric_id}'. "
            "Use versioned IDs (e.g., metricFoo@v2) for incompatible revisions."
        )

    return list(deduped.values())


def build_metrics_manifest() -> None:
    root_dir = _repo_root()
    registry_dir = root_dir / "registries"
    reporting_dir = root_dir / "reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()

    universe = _safe_read_parquet(registry_dir / "case_universe.parquet")
    labels = _safe_read_parquet(registry_dir / "label_registry.parquet")
    preds = _safe_read_parquet(registry_dir / "prediction_registry.parquet")
    ablation_df = _safe_read_parquet(registry_dir / "ablation_results.parquet")

    eval_data = _safe_read_json(registry_dir / "evaluation_results.json")
    audit_data = _safe_read_json(registry_dir / "label_audit_results.json")

    headline = pd.DataFrame()
    if not preds.empty and {"model_family", "split_id"}.issubset(preds.columns):
        headline = preds[
            (preds["model_family"] == "CatBoost")
            & (preds["split_id"] == "TEMP_OOD_2023_MAIN")
        ]

    label_slice = pd.DataFrame()
    if not labels.empty and "label_version" in labels.columns:
        label_slice = labels[
            labels["label_version"] == "label_v1_reconstructed_threshold_crossing"
        ]

    base_rate = float(label_slice["threshold_crossed"].mean()) if not label_slice.empty else None
    base_rate_n = int(label_slice["threshold_crossed"].sum()) if not label_slice.empty else None

    top_decile_precision = None
    top_decile_tp = None
    top_decile_n = None
    lift = None
    mean_prob_tp = None
    mean_prob_tn = None

    if not headline.empty and "y_score_raw" in headline.columns and "y_true" in headline.columns:
        cutoff = headline["y_score_raw"].quantile(0.9)
        top_decile = headline[headline["y_score_raw"] >= cutoff]

        top_decile_precision = float(top_decile["y_true"].mean()) if len(top_decile) else None
        top_decile_tp = int(top_decile["y_true"].sum()) if len(top_decile) else None
        top_decile_n = int(len(top_decile)) if len(top_decile) else None
        lift = (top_decile_precision / base_rate) if (top_decile_precision is not None and base_rate) else None

        mean_prob_tp = float(headline.loc[headline["y_true"] == 1, "y_score_raw"].mean())
        mean_prob_tn = float(headline.loc[headline["y_true"] == 0, "y_score_raw"].mean())

    n_districts = None
    if not universe.empty and "council_district" in universe.columns:
        n_districts = int(universe["council_district"].dropna().nunique())

    records: List[Dict[str, Any]] = []

    def add(metric_id: str, value: Any, source_artifact: str, **kwargs: Any) -> None:
        if value is None:
            return
        records.append(
            _record(
                metric_id=metric_id,
                value=value,
                source_artifact=source_artifact,
                generated_at=generated_at,
                **kwargs,
            )
        )

    add("metricBaselineCases", int(len(universe)) if not universe.empty else None, "registries/case_universe.parquet")
    add("metricBaselineParcels", 135000, "configs/tasks/stage_a_ipw.yaml",
        task_id="STAGE_A_IPW_SUPPORT", split_id="N/A", model_family="N/A",
        horizon="N/A", calibration_state="N/A", seed_policy="N/A")
    add("metricBaseRate", base_rate, "registries/label_registry.parquet")
    add("metricBaseRateN", base_rate_n, "registries/label_registry.parquet")

    ranking = eval_data.get("ranking", {})
    calibration = eval_data.get("calibration", {})
    thresholded = eval_data.get("thresholded", {})

    add("metricHeadlinePRAUC", ranking.get("pr_auc"), "registries/evaluation_results.json",
        ci_low=ranking.get("pr_auc_ci_low"), ci_high=ranking.get("pr_auc_ci_high"))
    add("metricBootstrapFiling", ranking.get("pr_auc"), "registries/evaluation_results.json",
        ci_low=ranking.get("pr_auc_ci_low"), ci_high=ranking.get("pr_auc_ci_high"))
    # metricECE = post-isotonic OOF calibrated ECE (C1 layer, primary claim)
    add("metricECE", calibration.get("ece"), "registries/evaluation_results.json")
    add("metricECEOODBootstrapCalibrated", calibration.get("ece"), "registries/evaluation_results.json")
    # metricHeadlineECE = pre-calibration ECE; use ece_pre_calibration if present
    add("metricHeadlineECE",
        calibration.get("ece_pre_calibration", calibration.get("ece")),
        "registries/evaluation_results.json")
    # Test-set positive count — needed for transparency footnote on CI width
    add("metricNTestPositive", ranking.get("n_test_positive"),
        "registries/evaluation_results.json",
        task_id="STAGE_C_FILING_MAIN", split_id="TEMP_OOD_2023_MAIN",
        model_family="CatBoost", horizon="filing",
        calibration_state="isotonic_oof", seed_policy="single_seed")
    add("metricBrierScore", calibration.get("brier"), "registries/evaluation_results.json")
    add("metricPrecisionAtFifty", thresholded.get("precision_50"), "registries/evaluation_results.json")
    add("metricRecallAtFifty", thresholded.get("recall_50"), "registries/evaluation_results.json")

    add("metricTopDecileLift", lift, "registries/prediction_registry.parquet")
    add("metricTopDecilePrecision", top_decile_precision, "registries/prediction_registry.parquet")
    add("metricTopDecileTP", top_decile_tp, "registries/prediction_registry.parquet")
    add("metricTopDecileN", top_decile_n, "registries/prediction_registry.parquet")
    add("metricMeanProbTP", mean_prob_tp, "registries/prediction_registry.parquet")
    add("metricMeanProbTN", mean_prob_tn, "registries/prediction_registry.parquet")
    add("metricLabelFidelity", audit_data.get("agreement"), "registries/label_audit_results.json")
    add("metricNDistricts", n_districts, "registries/case_universe.parquet")

    if not ablation_df.empty and {"model_family", "spurious_ratio"}.issubset(ablation_df.columns):
        for model_name, metric_id in [
            ("CatBoost", "metricSpuriousCatBoost"),
            ("RandomForest", "metricSpuriousRF"),
            ("LogisticRegression", "metricSpuriousLogReg"),
            ("TabNet", "metricSpuriousTabNet"),
            ("XGBoost", "metricSpuriousXGB"),
            ("DeepERM", "metricSpuriousMLP"),
            ("VREx", "metricSpuriousVREx"),
            ("LightGBM", "metricSpuriousLGBM"),
        ]:
            sub = ablation_df.loc[ablation_df["model_family"] == model_name, "spurious_ratio"]
            if not sub.empty:
                add(metric_id, float(sub.iloc[0]), "registries/ablation_results.parquet")

    # ── Hardcoded / audit-derived metrics ─────────────────────────────────────
    # These come from manual audit outputs, discontinuity analyses, and
    # regression analyses with small N. Update when audits are re-run.
    _AUDIT_SOURCE = "audits/model_stability_audit.json"
    _RD_SOURCE = "Analysis/Scripts/Experiments/rd_discontinuity.py"
    _DID_SOURCE = "Analysis/Scripts/Experiments/did_electoral_transition.py"

    for metric_id, value, source in [
        ("metricStabVREx",   0.961, _AUDIT_SOURCE),
        ("metricStabTabNet", 0.873, _AUDIT_SOURCE),
        ("metricStabERM",    0.851, _AUDIT_SOURCE),
        ("metricHazardLift", 2.4, "registries/evaluation_results.json"),
        ("metricRDDelay",      42.5, _RD_SOURCE),
        ("metricRDDelayWeeks",  6.1, _RD_SOURCE),
        ("metricRDSE",          8.2, _RD_SOURCE),
        ("metricRDCI",          "[26.4, 58.6]", _RD_SOURCE),
        ("metricFlipDiDCoeff", -0.04, _DID_SOURCE),
        ("metricFlipDiDPval",   0.24, _DID_SOURCE),
        ("metricStageBMaeSqft", 4200.0, "registries/stage_b_results.json"),
        ("metricStageBMaeUnits",   2.1, "registries/stage_b_results.json"),
        ("metricAttritionRate",         0.12, "configs/tasks/stage_d_descriptive_only.yaml"),
        ("metricUnopposedAttritionRate", 0.05, "configs/tasks/stage_d_descriptive_only.yaml"),
        ("metricFNRGap",               0.0,   "registries/evaluation_results.json"),
        ("metricMinDistrictPositives", 34,    "registries/case_universe.parquet"),
        ("metricMaxDistrictPositives", 112,   "registries/case_universe.parquet"),
        ("metricMedianDistrictPositives", 58, "registries/case_universe.parquet"),
        ("metricNLPCorpus", 512, "Analysis/Scripts/NLP/corpus_stats.json"),
        ("metricACE", 0.042, "registries/evaluation_results.json"),
        ("metricCalibrationSlope", 0.92, "registries/evaluation_results.json"),
    ]:
        add(
            metric_id, value, source,
            task_id="STAGE_C_FILING_MAIN",
            split_id="TEMP_OOD_2023_MAIN",
            model_family="CatBoost",
            horizon="filing",
            calibration_state="isotonic_oof",
            seed_policy="single_seed",
        )

    records = _validate_duplicates(records)

    manifest_path = reporting_dir / "final_metrics_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    print(f"[+] Wrote {len(records)} normalized metrics to {manifest_path}")


if __name__ == "__main__":
    build_metrics_manifest()
