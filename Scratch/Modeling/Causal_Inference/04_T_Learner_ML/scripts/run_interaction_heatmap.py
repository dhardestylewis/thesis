import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit
import shap
import warnings

warnings.filterwarnings('ignore')

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts"

FEATS_DYNAMIC = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count", "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age", "mortgage_rate_30yr", "mortgage_rate_30yr_momentum",
    "treasury_10yr_yield", "fed_funds_rate", "local_unemployment_rate",
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

def main():
    print("1. Loading Bi-Weekly Panel...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    df_hazard = df[df["period_seq"] > 0].copy()
    target = "petition_event"
    
    model_df = df_hazard[["case_number", target] + FEATS_DYNAMIC].copy().dropna()
    
    # Train model on 80% split
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))
    train = model_df.iloc[train_idx]
    
    X_train, y_train = train[FEATS_DYNAMIC], train[target]
    pos_count = y_train.sum()
    scale_pos_weight = (len(y_train) - pos_count) / pos_count if pos_count > 0 else 1.0
    
    print("2. Training CatBoostClassifier...")
    model = CatBoostClassifier(
        iterations=300, depth=6, scale_pos_weight=scale_pos_weight,
        random_seed=42, verbose=0, task_type="GPU"
    )
    model.fit(Pool(X_train, y_train))
    
    print("3. Executing TreeExplainer Interactions...")
    X_all = model_df[FEATS_DYNAMIC]
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_all)
    
    print("4. Computing Matrix and Scaling Alpha...")
    # Calculate absolute mean interaction effect (F x F)
    interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
    
    # Find top 12 features based on main effect (diagonal) to keep the matrix readable
    main_effects = np.diag(interaction_matrix)
    top_indices = np.argsort(main_effects)[-12:][::-1]
    
    top_features = [FEATS_DYNAMIC[i] for i in top_indices]
    sub_matrix = interaction_matrix[np.ix_(top_indices, top_indices)]
    
    # Isolate off-diagonal values to scale the transparency
    off_diag_mask = ~np.eye(sub_matrix.shape[0], dtype=bool)
    off_diag_max = sub_matrix[off_diag_mask].max()
    
    # Create an alpha mask based on interaction magnitude relative to the max interaction
    # The diagonal (main effects) will be forced to alpha=1.0
    # Add a minimum alpha (e.g., 0.1) so weak interactions are barely visible but not completely gone
    alpha_matrix = (sub_matrix / off_diag_max).clip(0, 1)
    alpha_matrix = np.where(off_diag_mask, 0.1 + 0.9 * alpha_matrix, 1.0)
    
    print("5. Rendering Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Plot using a vibrant colormap for interactions. We use the custom alpha matrix.
    # Seaborn doesn't natively support a 2D array for alpha, so we'll build it via matplotlib directly
    
    norm = Normalize(vmin=0, vmax=off_diag_max)
    cmap = plt.get_cmap("viridis")
    
    # Create RGBA colors for each cell
    rgba_colors = cmap(norm(sub_matrix))
    rgba_colors[..., 3] = alpha_matrix # Inject custom alpha
    
    # Plot the matrix using imshow
    ax.imshow(rgba_colors, aspect='auto', interpolation='nearest')
    
    # Add ticks and labels
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_yticks(np.arange(len(top_features)))
    
    # Beautify labels
    clean_labels = [f.replace('_', ' ').title() for f in top_features]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(clean_labels, fontsize=9)
    
    # Add numeric annotations (only for significant interactions)
    for i in range(len(top_features)):
        for j in range(len(top_features)):
            val = sub_matrix[i, j]
            if val > off_diag_max * 0.1 or i == j:
                color = "white" if alpha_matrix[i, j] < 0.5 or (i == j and val > off_diag_max * 1.5) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)
    
    # Add a colorbar (scaled to the off-diagonal interactions)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Absolute Interaction Magnitude (Log-Odds)', rotation=270, labelpad=15)
    
    plt.title("SHAP Feature Interaction Heatmap\n(Transparency scaled by interaction strength)", fontsize=14, weight="bold", y=1.05)
    plt.tight_layout()
    
    out_path = rf"{OUT_DIR}\causal_shap_interaction_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Heatmap saved to {out_path}")

if __name__ == "__main__":
    main()
