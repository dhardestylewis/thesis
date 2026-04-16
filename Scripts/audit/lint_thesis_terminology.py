#!/usr/bin/env python3
"""Fail when banned legacy petition terminology appears in .tex sources."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("valid petition", re.compile(r"\bvalid petition\b", re.IGNORECASE)),
    ("formal protest threshold", re.compile(r"\bformal protest threshold\b", re.IGNORECASE)),
    ("valid signed area percentage", re.compile(r"\bvalid signed area percentage\b", re.IGNORECASE)),
]

EXCLUDED_PARTS = {".git", "venv", ".venv", "node_modules"}
ALLOWED_PHRASES = {"clerk-certified valid petition (unobserved)"}


TARGET_DIRS = [ROOT / "Thesis_Draft" / "Draft_v1"]


def iter_tex_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGET_DIRS:
        if not target.exists():
            continue
        for path in target.rglob("*.tex"):
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            files.append(path)
    return files


def main() -> int:
    violations: list[str] = []

    for tex_file in iter_tex_files():
        rel = tex_file.relative_to(ROOT)
        for idx, line in enumerate(tex_file.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            normalized = line
            for allowed in ALLOWED_PHRASES:
                normalized = normalized.replace(allowed, "")
            for phrase, pattern in BANNED_PATTERNS:
                if pattern.search(normalized):
                    violations.append(f"{rel}:{idx}: banned phrase '{phrase}' -> {line.strip()}")

    if violations:
        print("Terminology lint failed. Replace banned phrases with controlled glossary terms.\n")
        print("\n".join(violations))
        return 1

    print("Terminology lint passed: no banned phrases found in .tex files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
