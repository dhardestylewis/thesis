import json
from pathlib import Path
import sys

# src/reporting/export_metrics_tex.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR

def export_metrics_tex():
    print("[+] Exporting Manifest to LaTeX (metrics_config.tex)...")
    
    manifest_path = REGISTRY_DIR / "metrics_manifest.json"
    tex_path = ROOT_DIR / "Thesis_Draft" / "Draft_v1" / "Tables" / "metrics_config.tex"
    
    if not manifest_path.exists():
        print(f"    [!] Error: Manifest {manifest_path} not found.")
        return
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    lines = ["% AUTO-GENERATED THESIS METRICS CONFIG", "% Source: registries/metrics_manifest.json\n"]
    
    for macro, entry in manifest.items():
        val = entry['value']
        if isinstance(val, str):
            val = val.replace("%", "\\%")
        lines.append(f"\\newcommand{{\\{macro}}}{{{val}}}")
        
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tex_path, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"    Exported {len(manifest)} macros to {tex_path}")

if __name__ == "__main__":
    export_metrics_tex()
