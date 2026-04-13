import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.reporting.build_metrics_manifest import build_metrics_manifest
from src.reporting.export_metrics_tex import _format_metric, export_metrics_tex


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: List[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gate_config(root: Path) -> Dict:
    default = {
        "headline_claim_metric_ids": ["metricBootstrapFiling", "metricECE", "metricTopDecileLift"],
        "required_glossary_terms": [
            "measured threshold-crossing petition",
            "out-of-distribution",
            "expected calibration error",
        ],
        "banned_phrases": [
            "TODO",
            "TBD",
            "lorem ipsum",
            "insert citation",
            "XX",
        ],
        "placeholder_patterns": [
            r"\{\{[^}]+\}\}",
            r"\[\[[^\]]+\]\]",
            r"<[^>]*placeholder[^>]*>",
        ],
        "manuscript_entrypoint": "Thesis_Draft/Draft_v1/Austin_NIMBY_Thesis_Draft.tex",
        "glossary_path": "Blei-Invariance_Causality-2026Spring/Knowledge_Base/Glossary.md",
    }
    override = _load_json(root / "reporting" / "submission_gate_config.json", {})
    default.update(override)
    return default


def _parse_tex_tree(entrypoint: Path) -> Tuple[Dict[Path, str], List[str]]:
    visited: Dict[Path, str] = {}
    errors: List[str] = []

    def walk(path: Path) -> None:
        if path in visited:
            return
        if not path.exists():
            errors.append(f"Missing input file: {path}")
            return
        text = path.read_text(encoding="utf-8", errors="ignore")
        visited[path] = text
        for rel in re.findall(r"\\input\{([^}]+)\}", text):
            next_path = (path.parent / rel)
            if next_path.suffix == "":
                next_path = next_path.with_suffix(".tex")
            walk(next_path.resolve())

    walk(entrypoint.resolve())
    return visited, errors


def _validate_glossary_terms(cfg: Dict, manuscript_text: str, root: Path) -> CheckResult:
    glossary_path = root / cfg["glossary_path"]
    details: List[str] = []
    if not glossary_path.exists():
        return CheckResult("glossary_terms", False, [f"Missing glossary: {glossary_path}"])

    glossary_text = glossary_path.read_text(encoding="utf-8", errors="ignore")
    terms = re.findall(r"^\*\*([^*\n]+)\*\*", glossary_text, flags=re.M)
    normalized = [t.strip().lower() for t in terms]
    dupes = sorted({t for t in normalized if normalized.count(t) > 1})
    if dupes:
        details.append(f"Duplicate glossary terms: {', '.join(dupes[:8])}")

    missing_required = [
        t for t in cfg["required_glossary_terms"] if t.lower() not in manuscript_text.lower()
    ]
    if missing_required:
        details.append("Missing required terms in manuscript: " + ", ".join(missing_required))

    return CheckResult("glossary_terms", not details, details)


def _check_banned_phrases(cfg: Dict, manuscript_text: str) -> CheckResult:
    details = []
    lowered = manuscript_text.lower()
    for phrase in cfg["banned_phrases"]:
        if phrase.lower() in lowered:
            details.append(f"Banned phrase present: '{phrase}'")
    return CheckResult("banned_phrases", not details, details)


def _check_placeholders(cfg: Dict, manuscript_text: str) -> CheckResult:
    details = []
    for pat in cfg["placeholder_patterns"]:
        matches = re.findall(pat, manuscript_text)
        if matches:
            snippet = ", ".join(matches[:5])
            details.append(f"Placeholder pattern matched ({pat}): {snippet}")
    return CheckResult("unresolved_placeholders", not details, details)


def _check_crossrefs(tex_files: Dict[Path, str], parse_errors: List[str]) -> CheckResult:
    details = list(parse_errors)
    labels: Set[str] = set()
    refd: Set[str] = set()
    dupes: Set[str] = set()

    for _, text in tex_files.items():
        for label in re.findall(r"\\label\{([^}]+)\}", text):
            if label in labels:
                dupes.add(label)
            labels.add(label)

        for cmd in ["ref", "autoref", "cref", "Cref"]:
            for blob in re.findall(rf"\\{cmd}\{{([^}}]+)\}}", text):
                for label in [x.strip() for x in blob.split(",")]:
                    refd.add(label)

        for env in ["figure", "table"]:
            for block in re.findall(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", text, flags=re.S):
                if "\\label{" not in block:
                    details.append(f"{env} block missing label")

    missing = sorted(label for label in refd if label not in labels)
    if missing:
        details.append("References without labels: " + ", ".join(missing[:15]))
    if dupes:
        details.append("Duplicate labels: " + ", ".join(sorted(dupes)[:15]))

    return CheckResult("cross_references", not details, details)


def _check_metric_integrity(root: Path, manuscript_text: str) -> CheckResult:
    manifest = _load_json(root / "reporting" / "final_metrics_manifest.json", [])
    macro_file = root / "Thesis_Draft" / "Draft_v1" / "Tables" / "metrics_config.tex"
    details: List[str] = []
    if not isinstance(manifest, list) or not manifest:
        return CheckResult("metric_integrity", False, ["Manifest missing or empty"])
    if not macro_file.exists():
        return CheckResult("metric_integrity", False, ["metrics_config.tex missing"])

    macro_text = macro_file.read_text(encoding="utf-8", errors="ignore")
    macro_map = {
        m.group(1): m.group(2)
        for m in re.finditer(r"\\newcommand\{\\([^}]+)\}\{([^}]*)\}", macro_text)
    }

    for row in manifest:
        metric_id = row["metric_id"]
        if metric_id in macro_map:
            expected = _format_metric(metric_id, row["value"]).replace("%", "\\%")
            if macro_map[metric_id] != expected:
                details.append(
                    f"Macro mismatch for {metric_id}: expected '{expected}', got '{macro_map[metric_id]}'"
                )

    used_macros = set(re.findall(r"\\(metric[A-Za-z0-9]+)\{\}", manuscript_text))
    undefined = sorted(m for m in used_macros if m not in macro_map)
    if undefined:
        details.append("Undefined metric macros in manuscript: " + ", ".join(undefined[:20]))

    return CheckResult("metric_integrity", not details, details)


def _emit_submission_report(root: Path, cfg: Dict) -> CheckResult:
    manifest = _load_json(root / "reporting" / "final_metrics_manifest.json", [])
    details: List[str] = []
    if not isinstance(manifest, list):
        return CheckResult("submission_report", False, ["Manifest invalid."])

    claim_ids = set(cfg["headline_claim_metric_ids"])
    claim_rows = [r for r in manifest if r.get("metric_id") in claim_ids]
    if not claim_rows:
        return CheckResult("submission_report", False, ["No headline claim metrics found in manifest."])

    report = {
        "headline_metrics": sorted({r["metric_id"] for r in claim_rows}),
        "task_ids": sorted({r.get("task_id", "") for r in claim_rows}),
        "split_ids": sorted({r.get("split_id", "") for r in claim_rows}),
        "model_families": sorted({r.get("model_family", "") for r in claim_rows}),
    }

    out_json = root / "reporting" / "submission_report.json"
    out_md = root / "reporting" / "submission_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# Submission Report", "", "## Headline Claims Coverage"]
    md.append("- Metrics: " + ", ".join(report["headline_metrics"]))
    md.append("- Task IDs: " + ", ".join(report["task_ids"]))
    md.append("- Split IDs: " + ", ".join(report["split_ids"]))
    md.append("- Model Families: " + ", ".join(report["model_families"]))
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    details.append(f"Wrote {out_json}")
    details.append(f"Wrote {out_md}")
    return CheckResult("submission_report", True, details)


def run_final_build_gate() -> int:
    root = _repo_root()
    cfg = _load_gate_config(root)

    print("[gate] building manifest")
    build_metrics_manifest()
    print("[gate] exporting TeX macros")
    export_metrics_tex()

    entrypoint = root / cfg["manuscript_entrypoint"]
    tex_files, parse_errors = _parse_tex_tree(entrypoint)
    combined_text = "\n".join(tex_files.values())

    checks = [
        _validate_glossary_terms(cfg, combined_text, root),
        _check_banned_phrases(cfg, combined_text),
        _check_placeholders(cfg, combined_text),
        _check_crossrefs(tex_files, parse_errors),
        _check_metric_integrity(root, combined_text),
        _emit_submission_report(root, cfg),
    ]

    failed = [c for c in checks if not c.ok]
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"[gate] {status}: {c.name}")
        for line in c.details:
            print(f"    - {line}")

    if failed:
        print("[gate] FINAL BUILD BLOCKED")
        return 1

    print("[gate] FINAL BUILD UNLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_final_build_gate())
