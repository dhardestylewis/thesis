import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
OUT_PLOT = rf"{OUT_DIR}\msm_propensity_overlap_fixed.png"

def main():
    print("1. Loading Biweekly Panel...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    # We rename for patsy formulas
    df = df.rename(columns={"archetype_pct_Spatial_Gravity": "pct_Spatial_Gravity"})
    
    features = [
        "land_acres", "proposed_max_height_ft", "pct_Spatial_Gravity", 
        "knn_petition_rate_1km", "dist_petition_rate_lag1", 
        "local_unemployment_rate", "mortgage_rate_30yr", "period_seq",
        "bw_sin", "bw_cos"
    ]
    
    # Fill NAs
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(0)
            
    # The treatment is whether a petition occurs in this specific biweekly period
    y_treat = df["petition_event"].fillna(0).astype(int)
    X = df[features]
    
    print("2. Training CatBoost Propensity Network...")
    # Train model to predict probability of treatment at Period t
    # MUST NOT use auto_class_weights='Balanced' for Propensity Scores, as it destroys the true probability
    prop_model = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4, verbose=0, random_seed=42)
    prop_model.fit(X, y_treat)
    
    # Extract probabilities
    p_treat = prop_model.predict_proba(X)[:, 1]
    
    # Clip probabilities to prevent infinite weights (using 1e-4 because marginal is ~0.001)
    p_treat = np.clip(p_treat, 1e-4, 0.99)
    
    print("3. Calculating Stabilized Inverse Probability Weights (IPTW)...")
    # Marginal probability of treatment (for stabilization)
    p_marginal = y_treat.mean()
    
    # Calculate Stabilized IPTW
    df["iptw"] = np.where(
        y_treat == 1,
        p_marginal / p_treat,
        (1 - p_marginal) / (1 - p_treat)
    )
    
    print("4. Executing Weighted Marginal Structural Model...")
    # Outcome: Bureaucratic Friction (Council Hearings this period)
    df["council_hearings_this_period"] = df["council_hearings_this_period"].fillna(0)
    
    # Unweighted Naive Regression
    naive_model = smf.ols("council_hearings_this_period ~ petition_event", data=df).fit(cov_type='cluster', cov_kwds={'groups': df['case_number']})
    naive_coef = naive_model.params["petition_event"]
    naive_se = naive_model.bse["petition_event"]
    
    # Weighted MSM Regression
    # statsmodels WLS (Weighted Least Squares) uses the weights argument
    msm_model = smf.wls("council_hearings_this_period ~ petition_event", data=df, weights=df["iptw"]).fit(cov_type='cluster', cov_kwds={'groups': df['case_number']})
    msm_coef = msm_model.params["petition_event"]
    msm_se = msm_model.bse["petition_event"]
    
    print(f"\n--- Results ---")
    print(f"Naive Effect: {naive_coef:.4f} (SE: {naive_se:.4f})")
    print(f"MSM Causal Effect: {msm_coef:.4f} (SE: {msm_se:.4f})")
    print("-----------------\n")
    
    print("5. Plotting Propensity Score Overlap...")
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Save p_treat to the dataframe for plotting
    df["p_treat"] = p_treat
    
    # Plot log-scaled propensity scores because the event is extremely rare (0.1% incidence)
    sns.kdeplot(df[df["petition_event"]==1]["p_treat"], label="Treated Group (Received Protest)", fill=True, color="#E74C3C", log_scale=True)
    sns.kdeplot(df[df["petition_event"]==0]["p_treat"], label="Control Group (No Protest)", fill=True, color="#3498DB", log_scale=True)
    
    plt.axvline(p_marginal, color='black', linestyle='--', label=f"Marginal Probability ({p_marginal:.4f})")
    plt.title("Marginal Structural Model: Propensity Score Overlap\n(CatBoost Treatment Predictions for Time-Varying Confounding)", fontsize=14, weight='bold')
    plt.xlabel("Predicted Probability of Protest at Period t (Log Scale)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, bbox_inches="tight")
    print(f"Master artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
