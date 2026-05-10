import os

ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\52e35f87-22e1-4135-9cf3-329ccde9b487'
PLOT_DIR = os.path.join(ARTIFACT_DIR, 'scratch', 'eda_plots')
HTML_PATH = os.path.join(ARTIFACT_DIR, 'eda_dashboard.html')

html = ['<html><head><style>body {font-family: Arial, sans-serif; padding: 20px;} img {max-width: 600px; margin: 10px; border: 1px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);} .plot-container {display: inline-block; margin-bottom: 30px;}</style></head><body>']
html.append('<h1>Comprehensive Causal Inference EDA Dashboard</h1>')

groups = {
    'Outcomes & Targets': ['Target_Height_Concession', 'resolved', 'censored', 'label_real_days_in_pipeline', 'Final_Zoning'],
    'PDF Features (Treatment & Targets)': ['pdf_requested_height_ft', 'pdf_reduced_to_ft', 'pdf_story_count', 'pdf_compatibility_height_ft'],
    'NLP Friction': ['nlp_document_count', 'nlp_oppose_hits', 'nlp_traffic_hits', 'nlp_density_hits', 'council_nlp_document_count', 'council_nlp_oppose_hits'],
    'Petition Dynamics': ['petition_pct_this_period', 'cumulative_petition_pct', 'cumulative_petition_events'],
    'Spatial Gravity': ['active_cases_100m', 'active_cases_500m', 'active_gravity_index_t', 'knn_petition_rate_1km', 'dist_petition_rate_lag1'],
    'Demographics (ACS)': ['total_population', 'median_household_income', 'renter_share', 'owner_share', 'median_age'],
    'Macro Context (FRED)': ['mortgage_rate_30yr', 'mortgage_rate_30yr_momentum', 'fed_funds_rate', 'local_unemployment_rate'],
    'Parcel Constraints': ['land_acres', 'yr_built', 'market_value', 'appraised_value']
}

for root, _, files in os.walk(PLOT_DIR):
    for group_name, variables in groups.items():
        html.append(f'<h2>{group_name}</h2>')
        for var in variables:
            img_name = f'plot_{var}.png'
            if img_name in files:
                html.append(f'<div class="plot-container"><h3>{var}</h3><img src="scratch/eda_plots/{img_name}" /></div>')

# Find any unmapped files
mapped = [f'plot_{var}.png' for vlist in groups.values() for var in vlist]
other_files = [f for f in files if f.endswith('.png') and f not in mapped]
if other_files:
    html.append('<h2>Other Variables</h2>')
    for f in other_files:
        html.append(f'<div class="plot-container"><h3>{f.replace("plot_", "").replace(".png", "")}</h3><img src="scratch/eda_plots/{f}" /></div>')

html.append('</body></html>')

with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))
