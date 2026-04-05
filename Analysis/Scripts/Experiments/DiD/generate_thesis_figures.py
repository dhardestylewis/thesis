import os
import pandas as pd
import matplotlib.pyplot as plt

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

import statsmodels.formula.api as smf
import geopandas as gpd
import contextily as cx

ROOT = r"C:\Users\dhl\data\thesis\thesis"
MASTER_DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
IMG_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Archive_Pipelines")
os.makedirs(IMG_DIR, exist_ok=True)

def main():
    print("[*] Generating Centralized Thesis Figures (Figures 1, 2, 3, OLS Table)...")
    
    if not os.path.exists(MASTER_DATA):
        print(f"[!] {MASTER_DATA} not found. Ensure pipeline is synchronized.")
        return
        
    df = pd.read_csv(MASTER_DATA, low_memory=False)
    
    # Standardize names from H0
    df['Year'] = pd.to_numeric(df['year'], errors='coerce')
    df['valid_petition'] = df['is_protested'].fillna(0).astype(int)
    
    # ---------------------------------------------------------
    # Render Figure 1: Spatial Map Temporal FacetGrid
    # ---------------------------------------------------------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        print("  -> Drawing Geodesic Mapping using strict H0 coordinates...")
        years_to_plot = sorted(df['Year'].dropna().unique())
        years_to_plot = [y for y in years_to_plot if y not in [2025, 2026]]
        
        # Calculate grid dynamically
        n_years = len(years_to_plot)
        cols = 6 if n_years >= 6 else n_years
        rows = (n_years + cols - 1) // cols
        
        fig1, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        df_geo = df.dropna(subset=['latitude', 'longitude', 'Year']).copy()
        gdf = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo.longitude, df_geo.latitude), crs="EPSG:4326")
        
        for idx, yr in enumerate(years_to_plot):
            ax = axes[idx]
            yr_gdf = gdf[gdf['Year'] == yr].to_crs(epsg=3857)
            
            unp = yr_gdf[yr_gdf['valid_petition'] == 0]
            if not unp.empty:
                unp.plot(ax=ax, color='blue', markersize=20, alpha=0.5)
                
            pro = yr_gdf[yr_gdf['valid_petition'] == 1]
            if not pro.empty:
                pro.plot(ax=ax, color='crimson', marker='X', markersize=80, edgecolor='darkred', linewidth=1.0)
                
            ax.set_title(f"H0 Master Cases: {yr}", fontsize=14, fontweight='bold')
            ax.set_axis_off()
            try: cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
            except: pass
            
        plt.suptitle("Multi-Year Temporal Dispersion of Authentic H0 Master Zoning Array", fontsize=24, y=1.02, fontweight='bold')
        plt.tight_layout()
        import matplotlib.lines as mlines
        blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Unprotested Variance')
        red_x = mlines.Line2D([], [], color='crimson', marker='X', linestyle='None', markersize=12, markeredgecolor='darkred', label='Valid Opposition')
        fig1.legend(handles=[blue_dot, red_x], loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=16)
        
        fig1.savefig(os.path.join(IMG_DIR, "fig1_spatial_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close(fig1)

    # ---------------------------------------------------------
    # Render Figure 2: Temporal Macro (Disabled - Orphaned from Thesis Draft)
    # ---------------------------------------------------------
    """
    print("  -> Drawing Figure 2 Timeseries directly from H0...")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    yearly_friction = df[df['valid_petition'] == 1].groupby('Year').size()
    ax2.bar(yearly_friction.index, yearly_friction.values, color='crimson', alpha=0.8, edgecolor='darkred')
    ax2.set_title('Empirical Dispersion of Valid Opposition (H0 Connected)', fontsize=14)
    ax2.set_ylabel('Valid Organizers', fontsize=12)
    fig2.savefig(os.path.join(IMG_DIR, "fig2_timeseries_macro.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    """

    # ---------------------------------------------------------
    # Render Figure 3: Demographic Friction (Disabled - Orphaned from Thesis Draft)
    # ---------------------------------------------------------
    """
    if 'acs_median_household_income' in df.columns:
        print("  -> Drawing demographic friction mappings...")
        import seaborn as sns
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=df[df['valid_petition'] == 0], x='acs_median_household_income', fill=True, color='gray', alpha=0.3, label='Unprotested', ax=ax3)
        sns.kdeplot(data=df[df['valid_petition'] == 1], x='acs_median_household_income', fill=True, color='crimson', alpha=0.5, label='Protested', ax=ax3)
        ax3.set_title("Density Distribution of Census Block Income Gap (H0)", fontsize=14)
        ax3.legend()
        fig3.savefig(os.path.join(IMG_DIR, "fig3_demographic_friction.png"), dpi=300, bbox_inches='tight')
        plt.close(fig3)
    """

    print("[+] Successfully re-aligned general thesis figures.")

if __name__ == "__main__":
    main()
