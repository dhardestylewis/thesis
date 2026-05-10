import pandas as pd
import numpy as np
import os

PANEL_PATH = r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv'
OUT_MD = r'C:\Users\dhl\.gemini\antigravity\brain\52e35f87-22e1-4135-9cf3-329ccde9b487\eda_all_features.md'

print("Loading panel...")
panel = pd.read_csv(PANEL_PATH, low_memory=False)

def format_describe(s):
    if s.dtype == 'object':
        desc = s.describe()
        return f"Count: {desc['count']:,} <br> Unique: {desc['unique']:,} <br> Top: {desc['top']}"
    else:
        desc = s.describe()
        return f"Mean: {desc['mean']:.2f} <br> Min: {desc['min']:.2f} <br> 50%: {desc['50%']:.2f} <br> Max: {desc['max']:.2f} <br> Non-Null: {desc['count']:,.0f}"

groups = {
    "Targets & Outcomes": [
        "resolved", "censored", "label_real_days_in_pipeline", "Final_Zoning", "Delta_Approved_Height"
    ],
    "PDF Height Concessions": [
        "pdf_requested_height_ft", "pdf_story_count", "pdf_reduced_to_ft", "pdf_compatibility_height_ft"
    ],
    "Cumulative NLP Friction": [
        "nlp_document_count", "nlp_oppose_hits", "nlp_traffic_hits", "nlp_density_hits",
        "council_nlp_document_count", "council_nlp_oppose_hits"
    ],
    "Petitions": [
        "petition_pct_this_period", "cumulative_petition_pct", "cumulative_petition_events"
    ],
    "Spatial Gravity & Active Stress": [
        "active_cases_100m", "active_cases_500m", "active_gravity_index_t", "knn_petition_rate_1km", "dist_petition_rate_lag1"
    ],
    "Parcel & EARS Baseline": [
        "market_value", "appraised_value", "land_acres", "yr_built", "improvement_sq_ft"
    ],
    "ACS Demographics": [
        "total_population", "median_household_income", "renter_share", "owner_share", "median_age"
    ],
    "FRED Macroeconomics": [
        "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "fed_funds_rate", "local_unemployment_rate"
    ]
}

md_lines = ["# Comprehensive EDA: Biweekly Panel Features & Targets\n"]
md_lines.append(f"**Total Rows:** {len(panel):,} | **Total Cases:** {panel['case_number'].nunique():,}\n")

# Calculate max/final values for cumulative features so we don't skew the mean with zeros from early periods
last_period = panel.sort_values("period_seq").groupby("case_number").tail(1)

for group_name, cols in groups.items():
    md_lines.append(f"## {group_name}")
    md_lines.append("| Feature | Missingness | Distribution / Stats |")
    md_lines.append("|---|---|---|")
    
    for c in cols:
        if c not in panel.columns:
            md_lines.append(f"| `{c}` | *Not found in panel* | - |")
            continue
            
        missing_pct = panel[c].isna().mean() * 100
        
        # For certain cumulative/static features, use last_period for descriptive stats
        is_cumulative_or_static = any(k in c for k in ['pdf_', 'nlp_', 'cumulative', 'Final', 'label_', 'appraised', 'median', 'population'])
        
        if is_cumulative_or_static:
            s = last_period[c].dropna()
        else:
            s = panel[c].dropna()
            
        if len(s) == 0:
            stats = "All Null"
        else:
            stats = format_describe(s)
            
        md_lines.append(f"| `{c}` | {missing_pct:.1f}% | {stats} |")
    md_lines.append("\n")

# Generate target specific EDA (e.g. days in pipeline distribution)
md_lines.append("## Target Profiling: `label_real_days_in_pipeline`\n")
md_lines.append("```text\n")
md_lines.append(last_period["label_real_days_in_pipeline"].describe().to_string())
md_lines.append("\n```\n")

os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"EDA Artifact generated successfully at: {OUT_MD}")
