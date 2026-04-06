import os
import re

file_path = r'c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Austin_NIMBY_Thesis_Draft.tex'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

replacements = [
    # 1. Against this backdrop, The central question is: can a valid protest petition filing be predicted before formal public hearings begin?
    (r'Against this backdrop,\s*the central empirical question is:\s*\\textbf\{can the filing of a valid protest petition against a housing development proposal be predicted before the project enters formal public hearings, using only information available at the time of filing\?\}',
     r'Against this backdrop, the central research question is whether a valid protest petition filing can be predicted before formal public hearings begin.'),
    (r'Against this backdrop, The central question is: can a valid protest petition filing be predicted before formal public hearings begin\?',
     r'Against this backdrop, the central research question is whether a valid protest petition filing can be predicted before formal public hearings begin.'),
     
    # 2. three-quarters three-fourths
    (r'three-quarters', r'three-fourths'),
    
    # force three-fourths council approval rule city council voting requirements
    (r'triggering a three-fourths supermajority requirement at City Council\.', r'requiring a three-fourths vote of the city council.'),
    (r'force three-fourths council approval rule city council voting requirements', r'require a three-fourths vote of the city council'),

    # This thesis asks whether a valid protest petition filing can be predicted before formal public hearings begin. The primary outcome is a directly observable, legally defined event approximated from public petition records and threshold calculations, invoking Austin\'s three-fourths council approval rule
    (r'asks whether a valid protest petition filing can be predicted before formal public hearings begin\. The primary outcome is a directly observable, legally defined event approximated from public petition records and threshold calculations, invoking Austin\'s three-fourths council approval rule', 
     r'asks whether a valid protest petition filing can be predicted before formal public hearings begin. The primary outcome is an operationalized petition-based measure intended to capture whether a case met the study-period petition threshold in reconstructed public records'),
    (r'The primary outcome is a legally defined, directly observable event: whether adjacent property owners file a petition meeting the 20\\% area threshold specified by Texas Local Government Code Chapter~211, thereby triggering a supermajority requirement at City Council\.',
     r'The primary outcome is an operationalized petition-based measure intended to capture whether a case met the study-period petition threshold in reconstructed public records.'),
     
    # directly observable, legally defined event
    (r'directly observable, legally defined event', r'operationalized outcome derived from petition filings and measured petition-share calculations'),
    
    # Stage B -- Project type classification performance
    (r'Stage B -- Project type classification performance', r'Stage B --- Project Type Classification'),
    (r'Project type classification performance Model Comparison', r'Project Type Classification Performance'),
    (r'Project Type and Scale Model Comparison\.', r'Project Type Classification Performance.'),

    # Predictive Architecture -> Modeling Strategy
    (r'Predictive Architecture', r'Modeling Strategy'),
    
    # Feature Libraries: Policy Forecasting vs. Explanatory Research
    (r'Feature Libraries: Policy Forecasting vs\\.\s*Explanatory Research', r'Predictive Covariates and Audit-Only Covariates'),
    (r'Feature Libraries: Policy Forecasting vs\.\s*Explanatory Research', r'Predictive Covariates and Audit-Only Covariates'),
    
    # support any administrative or planning application
    (r'support any practical planning use\.', r'support high-confidence administrative use.'),
    (r'support any administrative or planning application', r'support high-confidence administrative use'),
    
    # The thesis supplements its predictive framework with two causal analyses...
    (r'The thesis supplements its predictive framework with two causal analyses that examine the institutional mechanisms through which petitions affect outcomes\.', 
     r'The thesis also includes two exploratory quasi-experimental analyses of downstream institutional dynamics.'),
     
    # functions as a important constraint
    (r'functions as a important constraint', r'functions as an important constraint'),
    
    # The theoretical reality of participatory distortion only affects housing supply when...
    (r'The theoretical reality of participatory distortion only affects housing supply when it intersects with municipal institutions capable of stalling development\.', 
     r'Participatory distortion matters for housing supply when it operates through institutions that can delay or block projects.'),
     
    # crossing the legal 20% threshold introduces severe voting friction and procedural delay
    (r'crossing the legal 20\\%\s*threshold introduces severe voting friction and procedural delay\.', 
     r'cases exceeding the measured petition threshold appear to face additional voting friction and longer processing times.'),
     
    # strictly 'expressive' and no longer trigger
    (r'strictly ``expressive\'\' and no longer trigger the 3/4 supermajority vote requirement', 
     r'modified by HB 24; for certain noncomprehensive residential changes, adjoining-owner protests now require a higher threshold and carry a different vote rule'),

    # prediction policy problem
    (r'prediction policy problem', r'predictive policy question'),
    
    # Predicting which housing developments will face organized protest petitions is can be framed as such a problem.
    (r'Predicting which housing developments will face organized protest petitions is precisely such a problem\.', r'Predicting which zoning cases will meet the petition threshold can be framed as a predictive policy question.'),
    (r'Predicting which housing developments will face organized protest petitions is can be framed as such a problem\.', r'Predicting which zoning cases will meet the petition threshold can be framed as a predictive policy question.'),
    (r'is can be framed', r'can be framed'),
    
    # pro-housing policymakers
    (r'pro-housing policymakers', r'policymakers'),
    
    # algorithmic governance
    (r'algorithmic governance', r'administrative use of predictive models'),
    
    # This approach retains cases with partially missing covariates reduces artificial attrition
    (r'This data-retention approach reduces artificial attrition', r'This approach retains cases with partially missing covariates, reducing artificial attrition'),
    
    # A valid protest petition representing 20% ... triggers...
    (r'A valid protest petition representing 20\\%\s*of the area of lots immediately adjoining the proposed change triggers a supermajority requirement at City Council\.', 
     r'The figure schematically shows where a qualifying protest petition could make a three-fourths council vote relevant under the study-period framework. A valid protest petition representing 20\\% of the area of lots immediately adjoining the proposed change would require, under the applicable legal framework,'),
     
    # the probability of a valid protest petition filing precision-recall curves
    (r'Development hazard classification precision-recall curves', r'Precision--recall curves for the filing-date petition model'),
    (r'Opposition risk precision-recall curves by model architecture', r'Precision--recall curves for the filing-date petition model'),
    
    # best interpreted as a best interpreted as a ranking device
    (r'best interpreted as a best interpreted as a ranking device', r'best interpreted as a ranking device'),
    
    # The November 2022 council election is the primary primary period break.
    (r'The November 2022 council election is the primary regime boundary\.', r'The November 2022 council election is the primary period break in the analysis.'),
    
    # replaced a preservationist majority with a pro-housing three-fourths council approval rule
    (r'replaced a preservationist majority with a pro-housing supermajority during the 2023 to 2024 Watson-era council term', 
     r'produced a more pro-housing council majority during the 2023--2024 term'),
     
    # the probability of a valid protest petition filing Model: Filing-Date Predictive Performance
    (r'Opposition Risk Model: Baseline Predictive Capacity', r'Filing-Date Predictive Performance for the Petition Model'),
    
    # The the probability of a valid protest petition filing model
    (r'The Opposition Risk model evaluated at the filing date', r'The filing-date petition model'),
    
    # This expected error restricts deployment within automated administrative use
    (r'This expected error restricts deployment within automated decision-making pipelines', r'This level of calibration error makes fully automated administrative use difficult to justify'),
    
    # Global Model Calibration
    (r'Global Model Calibration Reliability and Capture Curves', r'Calibration of the Stage A and Stage C Models'),
    (r'Global Model Calibration:', r'Calibration of the Stage A and Stage C Models:'),
    
    # Out-of-Distribution Predictive Performance Decay
    (r'Out-of-Distribution Predictive Performance Decay', r'Performance Across Temporal Holdouts'),
    (r'Out-of-Distribution Performance Tracking', r'Performance Across Temporal Holdouts'),
    
    # demonstrating a district-level decline in valid petition probability
    (r'demonstrating a district-level decline in valid petition probability', r'showing the estimated interaction effect for treated districts'),
    
    # Three-Fourths council approval rule election
    (r'2022 Electoral Transition Outcomes', r'Estimated 2022 Electoral-Transition Interaction Effects by District'),
    
    # The a complementary descriptive check
    (r'The a complementary descriptive check', r'A complementary descriptive check'),
    
    # definitive policy forecasting
    (r'definitive policy forecasting', r'high-confidence case-level probability use'),
    
    # under class imbalance under class imbalance
    (r'under class imbalance under class imbalance', r'under class imbalance'),
    
    # Other edits
    (r'creates a the three-fourths', r'creates a three-fourths'),
    (r'an explicit, change', r'an explicit change'),
    (r'across all generic single-family residential base zones across all generic single-family residential base zones', r'across all generic single-family residential base zones'),
    (r'pro-housing three-fourths', r'three-fourths'),
    
    # Captions
    (r'Valid Petition Incidence by Proposed Land Use in Austin \\(Mercatus Center\\)', r'Reported Petition Incidence by Proposed Land Use in Austin'),
    (r'Austin Municipal Zoning Process Diagram', r'Schematic of the Austin Zoning Process'),
    (r'Development Occurrence Nested Targets', r'Stage A Target Definitions'),
    (r'Predicted Hotspot Density vs\.\\\s*Realized Events', r'High Predicted Development Risk and Observed Development Cases'),
    (r'Precision-Recall Area Under Curve \\(PR-AUC\\) Evaluation', r'PR-AUC for Filing-Date Petition Models'),
    (r'Expected Units in Projects Facing Valid Protest Petitions', r'Composite Measure of Expected Petitioned Units'),
    (r'Calibration and capture curves for the Stage A and Stage C models', r'Calibration and Capture Curves for the Stage A and Stage C Models'),
    (r'Temporal and policy-period holdout performance', r'Performance in Temporal and Policy-Period Holdouts'),
    (r'Grouped predictor importance for the filing-date model', r'Grouped Predictor Importance in the Filing-Date Petition Model'),
    (r'SHAP summary plot for the filing-date petition model', r'SHAP Summary for the Filing-Date Petition Model'),
    (r'Regression Discontinuity at the 20\\% Threshold', r'Threshold-Based Discontinuity at the Measured 20\\% Petition Share'),
    (r'Protest Petition Sharp Regression Discontinuity', r'Threshold-Based Discontinuity at the Measured 20\\% Petition Share'),
    (r'Methodological Causal Graphs', r'Identification Diagrams for the Threshold and HOME Analyses'),
    (r'Event Studies: HOME Initiative and HB~24', r'Exploratory Policy-Change Analyses: HOME and HB 24'),
    (r'2022 Electoral Transition District Outcomes', r'Estimated 2022 Electoral-Transition Interaction Effects by District'),
    (r'Multi-Period Placebo DiD Matrix', r'Placebo Estimates Across Earlier Election Cycles'),
    (r'Consolidated Geographic Causal Estimates', r'Summary of Supplementary Quasi-Experimental Estimates'),
    (r'Invariant Causal Prediction \\(ICP\\)', r'Supplementary Invariance Tests'),
    (r'Calibration Method Comparison', r'Supplementary Calibration Comparison'),
    (r'Research Project Overview', r'Interview Recruitment Overview'),
    (r'Temporal Panel Data Structure', r'Illustrative Support-Layer Panel Structure'),
    (r'Hyperparameter search results for the Stage A and Stage C models', r'Supplementary Hyperparameter Search Results'),
    (r'Grid Search PR-AUC Optimization Heatmaps', r'Supplementary Hyperparameter Search Results'),

    # More terminology replacements from custom fixes
    (r'architectures', r'models'),
    (r'Architecture', r'Model'),
    (r'deployment', r'administrative application'),
    (r'hotspots', r'areas of high predicted incidence'),
    (r'hotspot', r'areas of high predicted incidence'),
    (r'decision support', r'case-level use'),
    (r'orchestrator', r'specification'),
    (r'out-of-distribution tracking', r'evaluating temporal holdouts'),
]

for old, new in replacements:
    text = re.sub(old, new, text)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Replacement complete.")
