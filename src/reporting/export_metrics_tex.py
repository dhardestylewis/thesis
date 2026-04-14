import json
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_metric(metric_id: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
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
    tex_path = root / "Thesis_Draft" / "Draft_v1" / "Tables" / "metrics_config.tex"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if not isinstance(manifest, list):
        raise ValueError("Normalized manifest must be a JSON list of metric records.")

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

    lines = [
        "% AUTO-GENERATED THESIS METRICS CONFIG",
        "% Sources: reporting/final_metrics_manifest.json (computed)",
        "%          registries/metrics_manifest.json (golden/supplemental)",
        "% Computed values override golden where both are present.",
        "% CI for metricBootstrapFiling corrected to bootstrap-computed [0.64,0.95]",
        "% (n=2000 resamples, n_test_pos=15). Prior hardcoded [0.78,0.84] was incorrect.",
        "% ECE corrected: pre-calibration ECE=0.208; post-OOF-isotonic ECE=0.205.",
        "",
    ]
    for macro in sorted(macro_values):
        lines.append(f"\\newcommand{{\\{macro}}}{{{macro_values[macro]}}}")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] Exported {len(macro_values)} macros to {tex_path}")


if __name__ == "__main__":
    export_metrics_tex()
