import re

filepath = 'Austin_NIMBY_Thesis_Draft.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Define the optimal short titles mapped sequentially to the figures
optimal_titles = [
    "Context: Spatial Distribution of Zoning Cases (2007-2024)",
    "Context: 200ft Parcel Buffer Geometries for Spatial Targets",
    "Context: Austin Municipal Zoning Process Diagram",
    "Stage A: Development Hazard Classification PR Curves",
    "Stage A: Predicted Hotspot Density vs. Realized Events",
    "Stage C: Opposition Risk PR Curves by Model Architecture",
    "Stage F: Joint Expected Petition Probability Spatial Distribution",
    "Stage C: Global Model Calibration Reliability and Capture Curves",
    "Stage C: Out-of-Distribution Predictive Performance Decay",
    "Pipeline Engineering: Hierarchical Correlation Clustering of Top Predictors",
    "Stage C: SHAP Beeswarm Plot for the Opposition Model",
    "Causal Identification: Methodological Causal Graphs",
    "Causal Identification: Regression Discontinuity at the 20\\% Threshold",
    "Causal Identification: HOME Phase 1 Event-Study Coefficients",
    "Causal Identification: 2022 Electoral Transition District Outcomes",
    "Stage C: Geographic Distribution of False Positives vs. Actual Filings",
    "Pipeline Engineering: Grid Search PR-AUC Optimization Heatmaps",
    "NLP Framing: Mean Predicted Probability of Argument Frames by District",
    "NLP Framing: Argument Frame Probabilities: Opposed vs. Unopposed Cases"
]

def replace_figure_caption(match):
    global caption_index
    block = match.group(0)
    
    caption_match = re.search(r'\\caption\[(.*?)\]\{(.*?)\}', block, re.DOTALL)
    if caption_match and caption_index < len(optimal_titles):
        new_short = optimal_titles[caption_index]
        long_cap = caption_match.group(2)
        
        # We also want to prepend the Long Caption so it matches the Short Title visually in the body!
        # Wait, if we prepend it, it will show up boldly in the text.
        # Yes, let's make the long caption start with the bolded stage tag.
        # Actually, let's just make the long caption match the structure: \textbf{Stage X:} Long Caption.
        
        # First, strip existing bold tags at the very beginning of long caption to prevent duplicating them.
        stripped_long = re.sub(r'^\\textbf\{.*?\}:\s*', '', long_cap).strip()
        
        # Split the new short title to extract the tag (e.g. "Stage C:")
        parts = new_short.split(":", 1)
        tag = parts[0] + ":"
        
        final_long = f"\\textbf{{{tag}}} {stripped_long}"
        
        new_caption = f"\\caption[{new_short}]{{{final_long}}}"
        block = block.replace(caption_match.group(0), new_caption)
        caption_index += 1
        
    return block

caption_index = 0
text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', replace_figure_caption, text, flags=re.DOTALL)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Replaced {caption_index} figure titles.")
