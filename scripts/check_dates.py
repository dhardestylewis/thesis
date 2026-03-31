import os
import datetime
from pathlib import Path

base_dir = Path(r'c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1')
files = [
    r'Tables\multi_horizon_results.tex',
    r'summary_stats_table.tex',
    r'..\..\Analysis\Output\Archive_Pipelines\fig1_spatial_distribution.png',
    r'..\..\Analysis\Output\Exploratory_EDA\waller_buffer_map.png',
    r'..\..\Analysis\Output\Track0_Predictive\StageA_Figure3_PR_Curves.png',
    r'..\..\Analysis\Output\Track0_Predictive\StageA_Figure4_Hotspot.png',
    r'..\..\Analysis\Output\Chapter4\F22_Joint_Policy_Map.png',
    r'..\..\Analysis\Output\Archive_Pipelines\Fig8_Rolling_Origin_Horizons.png',
    r'..\..\Analysis\Output\Archive_Pipelines\Fig9_Model_Comparison_PR_AUC.png',
    r'..\..\Analysis\Output\Track1_Exhibits\fig_calibration_ece.pdf',
    r'..\..\Analysis\Output\Track1_Exhibits\fig_temporal_drift.pdf',
    r'..\..\Analysis\Output\Track1_Exhibits\fig_policy_regimes.pdf',
    r'..\..\Analysis\Output\Track1_Exhibits\fig_feature_importance.pdf',
    r'..\..\Analysis\Output\Chapter4\F8_Calibration.png',
    r'..\..\Analysis\Output\Chapter5\F16_Petition_RD.png',
    r'..\..\Analysis\Output\Chapter5\F17_HOME_EventStudy.png',
    r'..\..\Analysis\Output\Chapter6\F20_Stakeholder_Heatmap.png',
    r'..\..\Analysis\Output\Chapter6\F19_TextFrame_Composition.png',
    r'Qualitative_Appendix\project_overview.pdf',
    r'Qualitative_Appendix\protocol_summary.pdf',
    r'Qualitative_Appendix\interview_guide.pdf',
    r'Qualitative_Appendix\sample_protest_petition.pdf',
    r'..\..\Analysis\Output\Chapter4\F12_Opposition_PR.png'
]

print("--- FILE MODIFICATION TIMES ---")
for f in files:
    full_path = (base_dir / f).resolve()
    if full_path.exists():
        mtime = datetime.datetime.fromtimestamp(full_path.stat().st_mtime)
        print(f"{mtime.strftime('%Y-%m-%d %H:%M:%S')} | {full_path.name}")
    else:
        print(f"MISSING             | {f}")
print("-------------------------------")
