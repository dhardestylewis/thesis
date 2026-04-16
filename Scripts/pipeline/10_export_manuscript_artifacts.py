import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting.build_metrics_manifest import build_metrics_manifest
from src.reporting.export_metrics_tex import export_metrics_tex

if __name__ == "__main__":
    build_metrics_manifest()
    export_metrics_tex()
