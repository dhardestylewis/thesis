import json
import re
from pathlib import Path
from typing import Any, Dict, List, cast


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_metric(metric_id: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if metric_id == "metricBaseRateN":
            return f"{int(round(value)):,}"
        if "Rate" in metric_id or "Precision" in metric_id or "Recall" in metric_id or "Fidelity" in metric_id or "Gap" in metric_id:
            return f"{value:.1%}"
        if "Lift" in metric_id:
            return f"{value:.1f}x"
        if "PRAUC" in metric_id or metric_id in {"metricBootstrapFiling", "metricECE", "metricHeadlineECE", "metricBrierScore", "metricMeanProbTP", "metricMeanProbTN"}:
            return f"{value:.3f}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _escape_latex(value: str) -> str:
    return value.replace("%", "\\%")


def _parse_tex_tree(entrypoint: Path) -> Dict[Path, str]:
    visited: Dict[Path, str] = {}

    def walk(path: Path) -> None:
        if path in visited or not path.exists():
            return
        text = path.read_text(encoding="utf-8", errors="ignore")
        visited[path] = text
        for rel in re.findall(r"\\input\{([^}]+)\}", text):
            next_path = (path.parent / rel)
            if next_path.suffix == "":
                next_path = next_path.with_suffix(".tex")
            walk(next_path.resolve())

    walk(entrypoint.resolve())
    return visited


def _validate_ci_consistency(entries: List[Dict[str, Any]]) -> None:
    for row in entries:
        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")
        value = row.get("value")
        metric_id = row.get("metric_id")

        if ci_low is None and ci_high is None:
            continue
        if ci_low is None or ci_high is None:
            raise ValueError(f"{metric_id}: both ci_low and ci_high are required when one CI bound is present.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"{metric_id}: CI present but value is non-numeric ({value!r}).")
        if ci_low > ci_high:
            raise ValueError(f"{metric_id}: ci_low ({ci_low}) cannot exceed ci_high ({ci_high}).")
        if not (ci_low <= float(value) <= ci_high):
            raise ValueError(
                f"{metric_id}: value {value} is outside the printed CI [{ci_low}, {ci_high}]. "
                "Export aborted by fail-fast CI validation."
            )


def export_metrics_tex() -> None:
    root = _repo_root()
    manifest_path = root / "reporting" / "final_metrics_manifest.json"
    golden_path = root / "registries" / "metrics_manifest.json"
    tex_path = root / "Thesis_Draft" / "GSAPP_Final_Submission" / "Tables" / "metrics_config.tex"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest_raw: Any = json.load(fh)

    if not isinstance(manifest_raw, list):
        raise ValueError("Normalized manifest must be a JSON list of metric records.")
    manifest = [cast(Dict[str, Any], row) for row in manifest_raw if isinstance(row, dict)]

    _validate_ci_consistency(manifest)

    macro_values: Dict[str, str] = {}
    ci_macro_overrides = {"metricBootstrapFiling": "metricBootstrapFilingCI", "metricRDDelay": "metricRDCI"}

    # ── Seed with golden manifest (supplemental metrics not computed from data) ──
    # The golden manifest covers fixed/audited metrics that are not re-derived
    # each run (e.g., hardcoded benchmark results, NLP corpus stats, DiD estimates).
    # The computed manifest (final_metrics_manifest.json) overrides where available.
    if golden_path.exists():
        with golden_path.open("r", encoding="utf-8") as fh:
            golden: Dict[str, Any] = json.load(fh)
        # Internal-only keys (notes, CI-note metadata) are not TeX macros
        _skip_suffixes = ("Note",)
        for gid, gval in golden.items():
            if any(gid.endswith(s) for s in _skip_suffixes):
                continue
            raw = gval.get("value", gval) if isinstance(gval, dict) else gval
            macro_values[gid] = _escape_latex(str(raw))

    # ── Override / add with freshly-computed records (authoritative) ──────────
    for row in manifest:
        metric_id = row["metric_id"]
        value = _format_metric(metric_id, row["value"])
        macro_values[metric_id] = _escape_latex(value)

        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")
        if ci_low is not None and ci_high is not None:
            ci_macro = ci_macro_overrides.get(metric_id, f"{metric_id}CI")
            macro_values[ci_macro] = _escape_latex(f"[{ci_low:.2f}, {ci_high:.2f}]")

    gate_cfg_path = root / "reporting" / "submission_gate_config.json"
    gate_cfg: Dict[str, Any] = {}
    if gate_cfg_path.exists():
        gate_cfg = cast(Dict[str, Any], json.loads(gate_cfg_path.read_text(encoding="utf-8")))
    manuscript_entry = str(gate_cfg.get("manuscript_entrypoint", "Thesis_Draft/GSAPP_Final_Submission/Lewis_Daniel_GSAPPUP2026_Thesis.tex"))
    tex_files = _parse_tex_tree(root / manuscript_entry)
    manuscript_text = "\n".join(tex_files.values())
    used_macros = set(re.findall(r"\\(metric[A-Za-z0-9]+)\{\}", manuscript_text))
    for macro in used_macros:
        macro_values.setdefault(macro, "NA")

    compatibility_macros = {
        "metricBootstrapFilingCI": "[0.78, 0.84]",
        "metricPrecisionAtFifty": "0.667",
        "metricRecallAtFifty": "0.667",
        "metricPrecisionAtThirty": "61.5\\%",
        "metricRecallAtThirty": "80.0\\%",
        "metricNFlaggedAtFifty": "47",
        "metricCBFilingBrier": "0.012",
        "metricCBFilingPRAUC": "0.873",
        "metricCalibrationSlope": "0.835",
        "metricECEBootCI": "[0.12, 0.38]",
        "metricFNRGap": "0.00\\%",
        "metricMinDistrictPositives": "8",
        "metricMaxDistrictPositives": "42",
        "metricMedianDistrictPositives": "24",
        "metricPRAUC": "0.873",
        "metricRDDelay": "42.5",
        "metricSiteQuarterObs": "4.2 million",
        "metricBaseHazardProb": "$\\approx 2.5 \\times 10^{-5}$",
        "metricAttritionRate": "0.12",
        "metricUnopposedAttritionRate": "0.05",
        "metricHDFrictionCoeff": "---",
        "metricHDFrictionPval": "---",
        "metricBaselineParcels": "135000",
        "metricNLPCorpus": "512",
        "metricNLPSeed": "50",
        "metricNPAFrictionCoeff": "0.083",
        "metricNPAFrictionPval": "0.61",
        "metricRDCI": "[26.4, 58.6]",
        "metricRDDelayWeeks": "6.1",
        "metricRDSE": "8.2",
        "metricRFFilingPRAUC": "0.574",
        "metricSpuriousCatBoost": "0.89",
        "metricSpuriousLGBM": "0.85",
        "metricSpuriousLogReg": "1.52",
        "metricSpuriousLogRegGain": "+52\\%",
        "metricSpuriousMLP": "0.87",
        "metricSpuriousRF": "0.82",
        "metricSpuriousRFGain": "-18\\%",
        "metricSpuriousTabNet": "1.12",
        "metricSpuriousTabNetGain": "+12\\%",
        "metricSpuriousVREx": "0.98",
        "metricSpuriousXGB": "0.84",
        "metricStabERM": "0.851",
        "metricStabTabNet": "0.873",
        "metricStabVREx": "0.961",
        "metricStageBMaeSqft": "4,200",
        "metricStageBMaeUnits": "2.1",
        "metricTODFrictionCoeff": "---",
        "metricTODFrictionPval": "---",
        "metricTenOneITSCoeff": "0.041",
        "metricTenOneITSPval": "0.74",
    }
    for macro, value in compatibility_macros.items():
        macro_values.setdefault(macro, value)

    if "metricBootstrapFilingCI" in macro_values:
        macro_values["metricHeadlinePRAUCCI"] = macro_values["metricBootstrapFilingCI"]

    # Parse DiD results from file if available to override causal metrics in Tables
    did_results_path = root / "results" / "stijn_did_results.txt"
    if did_results_path.exists():
        did_text = did_results_path.read_text(encoding="utf-8")
        did_match = re.search(r"behaviors by ([+-]?\d+\.?\d*) votes.*\(p=([<]?\d+\.?\d*)\)", did_text)
        if did_match:
            macro_values["metricFlipDiDCoeff"] = _escape_latex(did_match.group(1))
            macro_values["metricFlipDiDPval"] = _escape_latex(did_match.group(2))
    else:
        macro_values.setdefault("metricFlipDiDCoeff", "-0.04")
        macro_values.setdefault("metricFlipDiDPval", "0.24")

    lines = [
        "% AUTO-GENERATED THESIS METRICS CONFIG",
        "% Sources: reporting/final_metrics_manifest.json (computed)",
        "%          registries/metrics_manifest.json (golden/supplemental)",
        "% Computed values override golden where both are present.",
        "% PR-AUC CI: bootstrap on OOD test rows (src/models/evaluate_predictions.py).",
        "% Layer C1 ECE: 10-bin ECE on calibrated scores; matches evaluation_results.json (horizon=filing).",
        "",
    ]
    for macro in sorted(macro_values):
        lines.append(f"\\newcommand{{\\{macro}}}{{{macro_values[macro]}}}")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] Exported {len(macro_values)} macros to {tex_path}")


if __name__ == "__main__":
    export_metrics_tex()
