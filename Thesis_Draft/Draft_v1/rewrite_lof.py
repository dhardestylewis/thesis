import re

filepath = 'Austin_NIMBY_Thesis_Draft.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Define the optimal short titles mapped sequentially to the figures
optimal_titles = [
    "Spatial Distribution of Zoning Cases (2007-2024)",
    "200ft Parcel Buffer Geometries for Spatial Targets",
    "Austin Municipal Zoning Process Diagram",
    "Development Hazard Classification Precision-Recall Curves",
    "Predicted Hotspot Density vs. Realized Development Events",
    "Opposition Risk Precision-Recall Curves by Model Architecture",
    "Joint Expected Petition Probability Spatial Distribution",
    "Global Model Calibration Reliability and Capture Curves",
    "Out-of-Distribution Predictive Performance Decay",
    "Hierarchical Correlation Clustering of Top Predictors",
    "SHAP Beeswarm Plot for the Opposition Model",
    "Methodological Causal Inference Graphs",
    "Regression Discontinuity at the 20\\% Valid Petition Threshold",
    "HOME Phase 1 Event-Study Coefficients",
    "2022 Electoral Transition District Outcomes",
    "Geographic Distribution of False Positive Predictions vs. Actual Filings",
    "Grid Search Precision-Recall AUC Optimization Heatmaps",
    "Mean Predicted Probability of Argument Frames by Council District",
    "Argument Frame Probabilities: Opposed vs. Unopposed Cases"
]

# We will find all \caption[...]{...} and replace the bracketed part iteratively
def replace_caption(match):
    global caption_index
    long_caption = match.group(2)
    
    # Tables also use \caption. We need to make sure we are only doing this for figures.
    # Actually, the regex might catch tables. Let's just stick to figures.
    # Wait! Tables have captions too. But Table captions usually don't have the truncation issue.
    pass

# Better approach: find \begin{figure} ... \end{figure} blocks and replace their captions.
def replace_figure_caption(match):
    global caption_index
    block = match.group(0)
    
    caption_match = re.search(r'\\caption\[(.*?)\]\{(.*?)\}', block, re.DOTALL)
    if caption_match and caption_index < len(optimal_titles):
        new_short = optimal_titles[caption_index]
        long_cap = caption_match.group(2)
        
        new_caption = f"\\caption[{new_short}]{{{long_cap}}}"
        block = block.replace(caption_match.group(0), new_caption)
        caption_index += 1
        
    return block

caption_index = 0
text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', replace_figure_caption, text, flags=re.DOTALL)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Replaced {caption_index} figure titles.")
