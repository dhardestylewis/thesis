"""
Master Inference Ladder
Executes Association -> Prediction -> Causation for key zoning outcomes.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
CASE_MASTER = r"C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv"
VOTES_TRANSCRIPT = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\zoning_cases_with_council_votes.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity_corrected.csv"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"

print("1. Assembling Data...")
# 1. Load Data
pet = pd.read_csv(PET_INTENSITY)
cm = pd.read_csv(CASE_MASTER, low_memory=False)
vt = pd.read_csv(VOTES_TRANSCRIPT, low_memory=False)
panel = pd.read_csv(PANEL_PATH, low_memory=False)
master = pd.read_csv(MASTER_PATH, low_memory=False)

# Standardize IDs
pet["case_number"] = pet["case_number"].str.strip()
cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
vt["Case_Number"] = vt["Case_Number"].str.strip()
master["case_number"] = master["case_number"].str.strip()

# 2. Extract Outcomes
# A. Status Outcomes
def clean_status(s):
    if pd.isna(s): return "Unknown"
    s = s.lower()
    if "withdrawn" in s: return "Withdrawn"
    if "denied" in s: return "Denied"
    if "closed" in s or "void" in s or "expired" in s: return "Passive_Death"
    if "approved" in s: return "Approved"
    return "Pending"

cm_status = cm[["CASE_NUMBER", "DETAILED_STATUS"]].drop_duplicates("CASE_NUMBER").copy()
cm_status["status_cat"] = cm_status["DETAILED_STATUS"].apply(clean_status)
# Let's fix the AMANDA "Closed" vs "Approved" quirk using master's Derived_Status
master_status = master[["case_number", "Derived_Status"]].drop_duplicates("case_number").copy()
cm_status = cm_status.merge(master_status, left_on="CASE_NUMBER", right_on="case_number", how="left")
cm_status.loc[(cm_status["status_cat"] == "Passive_Death") & (cm_status["Derived_Status"].str.contains("Approved", na=False)), "status_cat"] = "Approved"

cm_status["t_withdrawal"] = (cm_status["status_cat"] == "Withdrawn").astype(float)
cm_status["t_denial"] = (cm_status["status_cat"] == "Denied").astype(float)
cm_status["t_passive_death"] = (cm_status["status_cat"] == "Passive_Death").astype(float)
cm_status["t_approval"] = (cm_status["status_cat"] == "Approved").astype(float)

# B. Vote Splits
vote_pattern = re.compile(r'\b(\d{1,2})-(\d{1,2})\s*vote\b', re.IGNORECASE)
parsed_votes = []
for _, row in vt.iterrows():
    matches = vote_pattern.findall(str(row["Vote_Transcript"]))
    if matches:
        for m in matches:
            yes, no = int(m[0]), int(m[1])
            if 3 <= (yes + no) <= 11:
                parsed_votes.append({"Case_Number": row["Case_Number"], "no_votes": no})

vt_df = pd.DataFrame(parsed_votes)
if not vt_df.empty:
    vt_agg = vt_df.groupby("Case_Number").agg(t_max_nay_votes=("no_votes", "max")).reset_index()
else:
    vt_agg = pd.DataFrame(columns=["Case_Number", "t_max_nay_votes"])

# C. Bureaucratic Delay (Total Hearings)
hearings = panel.groupby("case_number").agg(
    t_total_council_hearings=("council_hearings_this_period", "sum")
).reset_index()

# D. Reductions/Concessions
OVERLAY_STRIP = __import__("re").compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)
master["req_int"] = master["Requested_Zoning"].apply(get_int)
master["fin_int"] = master["Final_Zoning"].apply(get_int)
master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)
concessions = master[["case_number", "t_downgrade"]].drop_duplicates("case_number")

# 3. Extract Covariates (Period 1 baseline)
base_cov = panel[panel["period_seq"] == 1][["case_number", "market_value", "median_household_income", "renter_share", "proposed_max_height_ft", "existing_max_height_ft"]].drop_duplicates("case_number").copy()
base_cov["market_value"] = np.log1p(base_cov["market_value"])
base_cov["median_household_income"] = np.log1p(base_cov["median_household_income"])
base_cov["height_delta"] = base_cov["proposed_max_height_ft"].fillna(0) - base_cov["existing_max_height_ft"].fillna(0)
base_cov = base_cov.fillna(base_cov.median(numeric_only=True))

# 4. Merge Master Dataset
df = cm_status[["CASE_NUMBER", "t_withdrawal", "t_denial", "t_passive_death", "t_approval"]].rename(columns={"CASE_NUMBER": "case_number"})
df = df.merge(vt_agg.rename(columns={"Case_Number": "case_number"}), on="case_number", how="left")
df["t_max_nay_votes"] = df["t_max_nay_votes"].fillna(0)
df = df.merge(hearings, on="case_number", how="left")
df = df.merge(concessions, on="case_number", how="left")
df = df.merge(pet[["case_number", "true_petition_pct", "petition_n_parcels"]], on="case_number", how="left")
df["true_petition_pct"] = df["true_petition_pct"].fillna(0)
df["any_protest"] = (df["petition_n_parcels"] > 0).astype(int)
df["running_var"] = df["true_petition_pct"] - 20
df = df.merge(base_cov, on="case_number", how="inner")

targets = ["t_withdrawal", "t_denial", "t_passive_death", "t_approval", "t_total_council_hearings", "t_max_nay_votes", "t_downgrade"]
covars = ["market_value", "median_household_income", "renter_share", "height_delta"]

results = []

print("2. Executing Inference Ladder...")
for t in targets:
    t_df = df.dropna(subset=[t]).copy()
    if len(t_df) < 50: continue
    
    # --- TIER 1: ASSOCIATION ---
    mean_no = t_df[t_df["any_protest"] == 0][t].mean()
    mean_yes = t_df[t_df["any_protest"] == 1][t].mean()
    if t_df[t].nunique() == 2:
        # Binary - Chi2
        ct = pd.crosstab(t_df["any_protest"], t_df[t])
        if ct.shape == (2,2):
            chi2, p_assoc, _, _ = stats.chi2_contingency(ct)
        else: p_assoc = np.nan
    else:
        # Continuous - Mann Whitney
        stat, p_assoc = stats.mannwhitneyu(t_df[t_df["any_protest"]==1][t], t_df[t_df["any_protest"]==0][t], alternative='two-sided')
        
    # --- TIER 2: PREDICTION (Regression) ---
    X = sm.add_constant(t_df[["any_protest"] + covars])
    y = t_df[t]
    try:
        if t_df[t].nunique() == 2:
            model = sm.Logit(y, X).fit(disp=0)
            coef_pred = model.params["any_protest"]
            p_pred = model.pvalues["any_protest"]
        else:
            model = sm.OLS(y, X).fit()
            coef_pred = model.params["any_protest"]
            p_pred = model.pvalues["any_protest"]
    except:
        coef_pred, p_pred = np.nan, np.nan
        
    # --- TIER 2b: CONTINUOUS DOSAGE (Among all cases) ---
    X_cont = sm.add_constant(t_df[["true_petition_pct"] + covars])
    try:
        if t_df[t].nunique() == 2:
            model_cont = sm.Logit(y, X_cont).fit(disp=0)
        else:
            model_cont = sm.OLS(y, X_cont).fit()
        coef_dosage = model_cont.params["true_petition_pct"]
        p_dosage = model_cont.pvalues["true_petition_pct"]
    except:
        coef_dosage, p_dosage = np.nan, np.nan
        
    # --- TIER 3a: CAUSATION (PSM) ---
    # 1:1 Nearest Neighbor matching on covariates
    scaler = StandardScaler()
    X_cov_scaled = scaler.fit_transform(t_df[covars])
    treated_idx = np.where(t_df["any_protest"] == 1)[0]
    control_idx = np.where(t_df["any_protest"] == 0)[0]
    
    if len(treated_idx) > 0 and len(control_idx) > 0:
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(X_cov_scaled[control_idx])
        distances, indices = nn.kneighbors(X_cov_scaled[treated_idx])
        matched_control_idx = control_idx[indices.flatten()]
        
        y_treated = t_df[t].iloc[treated_idx].values
        y_control_matched = t_df[t].iloc[matched_control_idx].values
        att_psm = np.mean(y_treated - y_control_matched)
        _, p_psm = stats.ttest_rel(y_treated, y_control_matched)
    else:
        att_psm, p_psm = np.nan, np.nan
        
    # --- TIER 3b: CAUSATION (RD at 20%) ---
    rd_df = t_df[(t_df["running_var"] >= -20) & (t_df["running_var"] <= 20)].copy()
    if len(rd_df[rd_df["running_var"] >= 0]) >= 10 and len(rd_df[rd_df["running_var"] < 0]) >= 10:
        left = rd_df[rd_df["running_var"] < 0][t].mean()
        right = rd_df[rd_df["running_var"] >= 0][t].mean()
        ate_rd = right - left
        pooled_se = np.sqrt(rd_df[rd_df["running_var"] < 0][t].var()/max(len(rd_df[rd_df["running_var"] < 0]),1) + rd_df[rd_df["running_var"] >= 0][t].var()/max(len(rd_df[rd_df["running_var"] >= 0]),1))
        z_stat = ate_rd / (pooled_se + 1e-9)
        p_rd = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        ate_rd, p_rd = np.nan, np.nan
        
    results.append({
        "Target": t.replace("t_", "").replace("_", " ").title(),
        "Base_NoProtest": mean_no,
        "Base_Protest": mean_yes,
        "Assoc_P": p_assoc,
        "Pred_Coef": coef_pred,
        "Pred_P": p_pred,
        "Dosage_Coef": coef_dosage,
        "Dosage_P": p_dosage,
        "PSM_ATT": att_psm,
        "PSM_P": p_psm,
        "RD_ATE_20bw": ate_rd,
        "RD_P": p_rd
    })

res_df = pd.DataFrame(results)

# Format output
def fmt_p(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

markdown = ["# Systematic Inference Ladder Results\n\n"]
markdown.append("| Target | Association (Raw Gap) | Binary Prediction | Continuous Dosage (Per 1%) | Causation (PSM ATT) | Causation (RD 20% ATE) |")
markdown.append("|---|---|---|---|---|---|")

for _, r in res_df.iterrows():
    gap = r['Base_Protest'] - r['Base_NoProtest']
    assoc_str = f"{gap:+.3f} ({fmt_p(r['Assoc_P'])})"
    pred_str = f"{r['Pred_Coef']:+.3f} ({fmt_p(r['Pred_P'])})" if not pd.isna(r['Pred_Coef']) else "-"
    dosage_str = f"{r['Dosage_Coef']:+.4f} ({fmt_p(r['Dosage_P'])})" if not pd.isna(r['Dosage_Coef']) else "-"
    psm_str = f"{r['PSM_ATT']:+.3f} ({fmt_p(r['PSM_P'])})" if not pd.isna(r['PSM_ATT']) else "-"
    rd_str = f"{r['RD_ATE_20bw']:+.3f} ({fmt_p(r['RD_P'])})" if not pd.isna(r['RD_ATE_20bw']) else "-"
    markdown.append(f"| **{r['Target']}** | {assoc_str} | {pred_str} | {dosage_str} | {psm_str} | {rd_str} |")

markdown.append("\n*Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant*\n")

markdown.append("### Interpretation Guide:\n")
markdown.append("- **Association:** Raw difference between protested and non-protested cases.\n")
markdown.append("- **Binary Prediction:** Is the *presence* of a protest an independent predictor after controlling for covariates?\n")
markdown.append("- **Continuous Dosage:** Does adding an extra 1% of protesting neighbors incrementally change the outcome, controlling for covariates?\n")
markdown.append("- **PSM (Propensity Score Matching):** Comparing protested cases to identically-matched non-protested cases (Isolating the selection effect).\n")
markdown.append("- **RD (Regression Discontinuity):** Does hitting the 20% legal supermajority threshold cause a discontinuous jump? (Isolating the legal mechanism).\n")


with open(rf"{OUT_DIR}\inference_ladder_results.md", "w") as f:
    f.write("\n".join(markdown))
    
print("Execution complete. Artifact saved.")
