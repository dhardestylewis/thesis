"""Verify \\includegraphics targets exist for the main thesis entrypoint.

Resolves paths using \\graphicspath entries in the manuscript root (same rules as
graphicx: each prefix is prepended to the filename; cwd is the main .tex directory).

Run from repo root:
  python scripts/verify_tex_assets.py
Exit code 1 if any path is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "Thesis_Draft" / "Draft_v1"
ENTRY = DRAFT / "Austin_NIMBY_Thesis_Draft.tex"

INCLUDE_GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
GRAPHICSPATH = re.compile(r"\\graphicspath\{((?:\{[^}]+\})+)\}")


def _parse_graphicspath(entry_text: str) -> list[str]:
    m = GRAPHICSPATH.search(entry_text)
    if not m:
        return []
    return re.findall(r"\{([^}]+)\}", m.group(1))


def _collect_tex_files(path: Path, seen: set[Path] | None = None) -> list[Path]:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen or not path.exists():
        return []
    seen.add(path)
    out = [path]
    text = path.read_text(encoding="utf-8", errors="ignore")
    for rel in re.findall(r"\\input\{([^}]+)\}", text):
        nxt = path.parent / rel
        if not nxt.suffix:
            nxt = nxt.with_suffix(".tex")
        out.extend(_collect_tex_files(nxt, seen))
    return out


def _resolve_graphics(rel: str, prefixes: list[str]) -> list[Path]:
    rel = rel.split(",")[0].strip()
    candidates: list[Path] = []
    candidates.append((DRAFT / rel).resolve())
    for pref in prefixes:
        candidates.append((DRAFT / pref / rel).resolve())
    return candidates


def main() -> None:
    entry_text = ENTRY.read_text(encoding="utf-8", errors="ignore")
    prefixes = _parse_graphicspath(entry_text)
    tex_files = _collect_tex_files(ENTRY)
    missing: list[tuple[str, str, list[str]]] = []

    for tex in tex_files:
        text = tex.read_text(encoding="utf-8", errors="ignore")
        for m in INCLUDE_GRAPHICS.finditer(text):
            raw = m.group(1).strip()
            cands = [str(p) for p in _resolve_graphics(raw, prefixes)]
            if not any(Path(p).exists() for p in cands):
                missing.append((str(tex.relative_to(ROOT)), raw, cands))

    print(f"Scanned {len(tex_files)} TeX files; \\graphicspath prefixes: {prefixes!r}")
    if missing:
        print(f"MISSING {len(missing)} asset(s):")
        for src, raw, cands in missing[:50]:
            print(f"  {src}: {{{raw}}}")
            for c in cands[:4]:
                print(f"      tried: {c}")
        if len(missing) > 50:
            print(f"  ... ({len(missing) - 50} more)")
        sys.exit(1)
    print("All includegraphics paths resolve (with graphicspath).")
    sys.exit(0)


if __name__ == "__main__":
    main()
