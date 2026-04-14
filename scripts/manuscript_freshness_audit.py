"""Cross-check manuscript TeX lines against git blame dates vs registry/report timestamps.

Flags lines that mention metrics/results and whose last commit predates the latest
artifact generation time (stale prose risk). Does not modify files.

Usage (from repo root):
  python scripts/manuscript_freshness_audit.py
  python scripts/manuscript_freshness_audit.py --json reporting/freshness_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Lines matching these patterns are candidates for staleness if blame is old.
STALE_HINT = re.compile(
    r"\\metric[A-Za-z]+|PR-AUC|PRAUC|ECE|calibrat|bootstrap|"
    r"0\.\d{2,}|registries/|evaluation_results|final_metrics_manifest",
    re.IGNORECASE,
)


def _parse_tex_inputs(entry: Path, visited: Set[Path] | None = None) -> List[Path]:
    if visited is None:
        visited = set()
    out: List[Path] = []
    p = entry.resolve()
    if p in visited or not p.exists():
        return out
    visited.add(p)
    out.append(p)
    text = p.read_text(encoding="utf-8", errors="ignore")
    for rel in re.findall(r"\\input\{([^}]+)\}", text):
        nxt = (p.parent / rel).with_suffix(".tex") if not rel.endswith(".tex") else (p.parent / rel)
        out.extend(_parse_tex_inputs(nxt, visited))
    return out


def _max_artifact_time(root: Path) -> datetime:
    """Latest generated_at / file mtime among key artifacts."""
    candidates: List[Path] = [
        root / "registries" / "evaluation_results.json",
        root / "reporting" / "final_metrics_manifest.json",
    ]
    best: datetime = datetime.fromtimestamp(0, tz=timezone.utc)
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "generated_at" in data:
            raw = str(data["generated_at"])
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                best = max(best, dt)
            except ValueError:
                pass
        elif isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and row.get("generated_at"):
                    raw = str(row["generated_at"])
                    try:
                        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        best = max(best, dt)
                    except ValueError:
                        pass
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        best = max(best, mtime)
    return best


def _git_blame_line_dates(path: Path, root: Path) -> List[Tuple[int, datetime]]:
    """Return (line_no, committer_date) for each line using git blame -w --line-porcelain."""
    rel = path.relative_to(root)
    proc = subprocess.run(
        ["git", "blame", "-w", "--line-porcelain", "--", str(rel)],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return []
    lines_out: List[Tuple[int, datetime]] = []
    current_date: datetime | None = None
    line_no = 0
    for line in proc.stdout.splitlines():
        if line.startswith("committer-time "):
            ts = int(line.split()[1])
            current_date = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif line.startswith("\t"):
            line_no += 1
            if current_date is not None:
                lines_out.append((line_no, current_date))
            current_date = None
    return lines_out


def run_audit(entrypoint: str, root: Path) -> Dict[str, Any]:
    ep = (root / entrypoint).resolve()
    tex_files = _parse_tex_inputs(ep)
    threshold = _max_artifact_time(root)
    stale: List[Dict[str, Any]] = []
    for tex in tex_files:
        blame_map = dict(_git_blame_line_dates(tex, root))
        text_lines = tex.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, content in enumerate(text_lines, start=1):
            dt = blame_map.get(i)
            if dt is None:
                continue
            # Calendar-day comparison: same-day edits are not flagged as stale relative to a same-day pipeline run.
            if dt.date() >= threshold.date():
                continue
            if not STALE_HINT.search(content):
                continue
            stale.append(
                {
                    "file": str(tex.relative_to(root)),
                    "line": i,
                    "blame_utc": dt.isoformat(),
                    "threshold_utc": threshold.isoformat(),
                    "snippet": content[:240],
                }
            )
    return {
        "manuscript_entrypoint": entrypoint,
        "artifact_threshold_utc": threshold.isoformat(),
        "files_scanned": [str(p.relative_to(root)) for p in tex_files],
        "stale_metricish_lines": stale,
        "stale_count": len(stale),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--entrypoint",
        default="Thesis_Draft/Draft_v1/Austin_NIMBY_Thesis_Draft.tex",
        help="Root .tex file (walks \\input closure)",
    )
    ap.add_argument("--json", help="Write JSON report to this path")
    args = ap.parse_args()
    report = run_audit(args.entrypoint, ROOT)
    print(
        f"Artifact threshold (newest registry/report stamp): {report['artifact_threshold_utc']}\n"
        f"Files scanned: {len(report['files_scanned'])}\n"
        f"Stale metric-ish lines (blame older than threshold): {report['stale_count']}"
    )
    for row in report["stale_metricish_lines"][:80]:
        print(f"  {row['file']}:{row['line']} blame<{row['blame_utc'][:10]}  {row['snippet'][:100]}...")
    if report["stale_count"] > 80:
        print(f"  ... ({report['stale_count'] - 80} more)")
    if args.json:
        outp = Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[+] Wrote {outp}")


if __name__ == "__main__":
    main()
