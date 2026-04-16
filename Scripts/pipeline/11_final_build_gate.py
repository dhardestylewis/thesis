import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reporting.final_build_gate import run_final_build_gate


if __name__ == "__main__":
    raise SystemExit(run_final_build_gate())
