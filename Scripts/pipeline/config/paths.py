from pathlib import Path

# Repository Root (Dynamically resolves to the `thesis/` folder regardless of where the script is run)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Central Data Lake
DATA_DIR = ROOT_DIR / "Data"

# High-Level Subdirectories
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FINAL_DIR = DATA_DIR / "final"

# Domain-Specific Legacy Directories (These will eventually move to raw/)
ZONING_CASES_DIR = DATA_DIR / "Zoning_Cases"
PROTEST_PETITIONS_DIR = DATA_DIR / "Protest_Petitions"
COMMISSION_PDFS_DIR = DATA_DIR / "Commission_PDFs"
COUNCIL_DOCS_DIR = DATA_DIR / "Council_Documents"
COA_OPEN_DATA_DIR = DATA_DIR / "CoA_Open_Data"
GIS_DIR = DATA_DIR / "GIS"
PANEL_DIR = DATA_DIR / "Panel"

# Output Directories
OUTPUTS_DIR = ROOT_DIR / "Outputs"
FORECASTING_ARTIFACTS_DIR = OUTPUTS_DIR / "Forecasting_Artifacts"
