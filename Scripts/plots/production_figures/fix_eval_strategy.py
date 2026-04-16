import sys
import os

pth = r'Thesis_Draft\Draft_v1\Lewis_2026_NIMBYism_Austin_Thesis.tex'
with open(pth, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace block 1 (Primary evaluation & Metric-object reconciliation)
block1 = r"""\textbf{Primary evaluation.} The out-of-distribution bootstrap PR-AUC at the filing-date horizon is the primary 
evaluation object, as it approximates forward-like administrative use. In-distribution 5-fold cross-validation is 
retained only as an optimistic upper-bound object. Because this is primarily a ranking exercise, emphasis remains on 
ranking discrimination (PR-AUC, top-decile lift), with calibration and thresholded diagnostics treated as supporting 
information rather than coequal headline estimands.
  
  \textbf{Metric-object reconciliation.} The manuscript reports multiple metric objects for different inferential 
purposes, and they are not treated as interchangeable: (i) the headline filing-date object is the out-of-sample 
bootstrap estimate at the filing date, (ii) the 20-seed mean characterizes within-sample stochastic variance, and 
(iii) in-distribution cross-validation provides an optimistic upper bound. A reconciliation manifest maps manuscript 
macros to object metadata (task, split, model, calibration state, and source artifact)."""

replacement1 = r"""\textbf{Primary evaluation.} Out-of-distribution temporal cross-validation at the filing-date horizon is the primary evaluation object, as it maps directly onto forward-like administrative forecasting constraints. Because identifying actionable neighborhood opposition risk is functionally a ranking exercise, the evaluation framework prioritizes ranking discrimination (specifically PR-AUC and cumulative lift) bounded sequentially across time. In-distribution cross-validation surfaces are retained strictly to document algorithmic fragility and parameter stability, NOT to model operational accuracy."""

# Normalize newlines to match safely
import re

def safe_replace(full_text, target, repl):
    target_pattern = re.escape(target).replace(r'\ ', r'\s+').replace(r'\n', r'\s+')
    return re.sub(target_pattern, repl, full_text, count=1)

text = safe_replace(text, block1, replacement1)

block2 = r"""\textbf{Methodological Integrity and Pre-training Leakage.} The use of foundation models like TabPFN introduces a 
risk of "future leakage" if Austin's open data was inadvertently included in the model's original training corpus. To 
shield against this contamination, the foundation benchmarks were conducted under a strict \textit{Zero-Shot 
In-Context} enclosure. The internal model weights remained frozen, and the model was only permitted to "observe" the 
historically bounded pre-2022 training set as a prompt-context before predicting post-2022 outcomes. This protocol 
ensures that the model's 0.54 PR-AUC success is a result of structural pattern recognition, not historical 
memorization."""

text = safe_replace(text, block2, "")

with open(pth, 'w', encoding='utf-8') as f:
    f.write(text)

print("SUCCESS")
