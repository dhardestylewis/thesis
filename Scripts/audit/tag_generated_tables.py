import re
import os

tables = [
    r'Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_09_multi_horizon_results.tex',
    r'Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_03_calibration_benchmark.tex',
    r'Thesis_Draft\Draft_v1\Tables\chapter4_performance\tbl_ch4_02_alternative_architectures.tex'
]

tags = {
    r'multi_horizon_results.tex': r'\\caption{\\textbf{Stage C:} Multi-Horizon Opposition Model Performance with 95\\% Bootstrap CIs}',
    r'calibration_benchmark.tex': r'\\caption{\\textbf{Stage C:} Calibration Method Comparison: ECE, ACE, and Brier Score Across Architectures}',
    r'alternative_architectures.tex': r'\\caption{\\textbf{Pipeline Engineering:} Stage A \\& Stage C Evaluated at Filing: Raw vs. Calibrated Candidate Architectures}'
}

for fp in tables:
    with open(fp, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Very robust replace: just wipe the existing \caption{...} entirely for these explicit files
    text = re.sub(r'\\caption\{.*?\}', tags[os.path.basename(fp)], text)
    
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(text)

print("Dynamic Python tables successfully tagged!")
