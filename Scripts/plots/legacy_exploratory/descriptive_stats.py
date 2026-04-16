"""
descriptive_stats.py
====================
Master descriptive statistics for thesis reader meeting (Mar 13).
Produces tables, time-series, and spatial data on Austin zoning protest petitions.

Outputs saved to: Analysis/Output/Descriptive/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

import os, json, warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = r"C:\Users\dhl\data\thesis\thesis"
DATA       = os.path.join(ROOT, "Data")
OUT_DIR    = os.path.join(ROOT, "Analysis", "Output", "Descriptive")
os.makedirs(OUT_DIR, exist_ok=True)

PET_SUMMARY = os.path.join(DATA, "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv")
PET_SIGNERS = os.path.join(DATA, "Protest_Petitions", "petition_signers_from_pdf.csv")
ENRICHED_ZD = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "enriched_zoning_data_full.csv")
MULTI_PARCEL= os.path.join(DATA, "Zoning_Cases", "Processed_Data", "multi_parcel_closed_2018_2025.csv")
PANEL_V3    = os.path.join(DATA, "Panel", "Output", "Property_Year_Panel_Enriched.csv")
PANEL_ENR   = os.path.join(DATA, "Panel", "Output", "Property_Year_Panel_Enriched.csv")

# ── styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.size':        11,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
})
COLORS = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#db2777']


# ═══════════════════════════════════════════════════════════════════════════
# 1. PETITION TRENDS — "Was the protest tool increasingly used?"
# ═══════════════════════════════════════════════════════════════════════════
def section1_petition_trends():
    print("\n=== 1. PETITION TRENDS ===")
    pet = pd.read_csv(PET_SUMMARY)
    pet = pet.dropna(subset=['year'])
    pet['year'] = pet['year'].astype(int)

    # yearly counts
    yearly = pet.groupby('year').agg(
        n_cases        = ('case_number', 'count'),
        total_parcels  = ('total_parcels', 'sum'),
        mean_signers   = ('signers', 'mean'),
        mean_signer_pct= ('signer_pct', 'mean'),
        median_signer_pct = ('signer_pct', 'median'),
    ).reset_index()

    yearly.to_csv(os.path.join(OUT_DIR, "petition_trends_by_year.csv"), index=False)
    print(yearly.to_string(index=False))

    # ── Figure 1: cases per year ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(yearly['year'], yearly['n_cases'], color=COLORS[0], alpha=0.85, edgecolor='white')
    ax.set_title('Protest Petition Cases per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Cases')
    # annotate
    for _, row in yearly.iterrows():
        ax.text(row['year'], row['n_cases'] + 0.3, str(int(row['n_cases'])),
                ha='center', va='bottom', fontsize=9)

    ax = axes[0, 1]
    ax.bar(yearly['year'], yearly['total_parcels'], color=COLORS[1], alpha=0.85, edgecolor='white')
    ax.set_title('Total Parcels Involved per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Parcels')

    ax = axes[1, 0]
    ax.plot(yearly['year'], yearly['mean_signer_pct'], 'o-', color=COLORS[2], linewidth=2)
    ax.fill_between(yearly['year'], 0, yearly['mean_signer_pct'], alpha=0.15, color=COLORS[2])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_title('Mean Signer Percentage per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Signer %')
    ax.legend()

    ax = axes[1, 1]
    ax.plot(yearly['year'], yearly['mean_signers'], 's-', color=COLORS[3], linewidth=2)
    ax.set_title('Mean Number of Signers per Case per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Signers')

    fig.suptitle('Austin Zoning Protest Petition Trends (2007–2024)', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_petition_trends.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved fig1_petition_trends.png")

    # Summary statistics
    summary = {
        'total_cases': len(pet),
        'year_range': f"{pet['year'].min()}-{pet['year'].max()}",
        'total_parcels_involved': int(pet['total_parcels'].sum()),
        'mean_signers_per_case': round(pet['signers'].mean(), 1),
        'median_signer_pct': round(pet['signer_pct'].median(), 1),
        'peak_year': int(yearly.loc[yearly['n_cases'].idxmax(), 'year']),
        'peak_year_cases': int(yearly['n_cases'].max()),
    }
    print(f"\n  Summary: {json.dumps(summary, indent=2)}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 2. DEVELOPMENT OUTCOMES — "Did the development happen? Was it delayed?"
# ═══════════════════════════════════════════════════════════════════════════
def section2_protest_vs_development():
    print("\n=== 2. PROTEST VS DEVELOPMENT OUTCOMES ===")
    # Use enriched zoning data which has approval_date and status_date
    zd = pd.read_csv(ENRICHED_ZD)
    pet = pd.read_csv(PET_SUMMARY)

    protested_cases = set(pet['case_number'].dropna().unique())
    zd['protested'] = zd['case_number'].isin(protested_cases).astype(int)

    # Parse dates
    for col in ['application_start_date', 'status_date', 'approval_date', 'final_date']:
        if col in zd.columns:
            zd[col] = pd.to_datetime(zd[col], errors='coerce')

    # Compute processing time
    if 'application_start_date' in zd.columns and 'approval_date' in zd.columns:
        zd['days_to_approval'] = (zd['approval_date'] - zd['application_start_date']).dt.days

    # Approval rates
    zd['approved'] = zd['approval_date'].notna().astype(int)
    for label, subset in [('ALL CASES', zd), ('PROTESTED', zd[zd['protested']==1]), ('NOT PROTESTED', zd[zd['protested']==0])]:
        n = len(subset)
        n_approved = subset['approved'].sum()
        pct = round(100*n_approved/n, 1) if n > 0 else 0
        print(f"  {label}: {n} cases, {n_approved} approved ({pct}%)")
        if 'days_to_approval' in subset.columns:
            approved_subset = subset[subset['days_to_approval'].notna()]
            if len(approved_subset) > 0:
                print(f"    Median days to approval: {approved_subset['days_to_approval'].median():.0f}")
                print(f"    Mean days to approval:   {approved_subset['days_to_approval'].mean():.0f}")

    # ── Figure 2: Protested vs Not protested bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Approval rates
    ax = axes[0]
    groups = ['Protested', 'Not Protested']
    prot = zd[zd['protested']==1]
    notp = zd[zd['protested']==0]
    rates = [prot['approved'].mean()*100, notp['approved'].mean()*100]
    bars = ax.bar(groups, rates, color=[COLORS[1], COLORS[0]], alpha=0.85, edgecolor='white', width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Approval Rate (%)')
    ax.set_title('Zoning Case Approval Rates')
    ax.set_ylim(0, 105)

    # Panel B: Time to approval
    ax = axes[1]
    if 'days_to_approval' in zd.columns:
        data_p = prot['days_to_approval'].dropna()
        data_n = notp['days_to_approval'].dropna()
        bp = ax.boxplot([data_p, data_n], labels=['Protested', 'Not Protested'],
                       patch_artist=True, widths=0.5,
                       boxprops=dict(alpha=0.85),
                       medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(COLORS[1])
        bp['boxes'][1].set_facecolor(COLORS[0])
        ax.set_ylabel('Days to Approval')
        ax.set_title('Processing Time: Protested vs. Not Protested')

    fig.suptitle('Zoning Protest Petitions and Development Outcomes', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_protest_outcomes.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved fig2_protest_outcomes.png")

    # Save table
    outcome_table = zd.groupby('protested').agg(
        n_cases         = ('case_number', 'count'),
        n_approved      = ('approved', 'sum'),
        approval_rate   = ('approved', 'mean'),
        median_days     = ('days_to_approval', 'median'),
        mean_days       = ('days_to_approval', 'mean'),
    ).reset_index()
    outcome_table['protested'] = outcome_table['protested'].map({0: 'Not Protested', 1: 'Protested'})
    outcome_table.to_csv(os.path.join(OUT_DIR, "protest_vs_outcome.csv"), index=False)
    print(f"  → Saved protest_vs_outcome.csv")
    print(outcome_table.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# 3. ZONING TYPE ANALYSIS — "Is there different resistance to office vs apartment?"
# ═══════════════════════════════════════════════════════════════════════════
def section3_zoning_type():
    print("\n=== 3. RESISTANCE BY ZONING/LAND USE TYPE ===")
    zd = pd.read_csv(ENRICHED_ZD)
    pet = pd.read_csv(PET_SUMMARY)
    protested_cases = set(pet['case_number'].dropna().unique())
    zd['protested'] = zd['case_number'].isin(protested_cases).astype(int)

    # By proposed land use
    if 'proposed_land_use' in zd.columns:
        lu_table = zd.groupby('proposed_land_use').agg(
            n_cases    = ('case_number', 'count'),
            n_protested= ('protested', 'sum'),
            protest_rate= ('protested', 'mean'),
        ).sort_values('n_cases', ascending=False)
        lu_table['protest_rate'] = (lu_table['protest_rate'] * 100).round(1)
        lu_table.to_csv(os.path.join(OUT_DIR, "resistance_by_land_use.csv"))
        print("\n  By Proposed Land Use:")
        print(lu_table.head(15).to_string())

    # By case type
    if 'case_type' in zd.columns:
        ct_table = zd.groupby('case_type').agg(
            n_cases    = ('case_number', 'count'),
            n_protested= ('protested', 'sum'),
            protest_rate= ('protested', 'mean'),
        ).sort_values('n_cases', ascending=False)
        ct_table['protest_rate'] = (ct_table['protest_rate'] * 100).round(1)
        ct_table.to_csv(os.path.join(OUT_DIR, "resistance_by_case_type.csv"))
        print("\n  By Case Type:")
        print(ct_table.to_string())

    # ── Figure 3: Protest rate by proposed land use (top 10) ──
    if 'proposed_land_use' in zd.columns:
        top_lu = lu_table[lu_table['n_cases'] >= 5].head(10).reset_index()
        if len(top_lu) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(top_lu['proposed_land_use'], top_lu['protest_rate'],
                          color=COLORS[4], alpha=0.85, edgecolor='white')
            for bar, n in zip(bars, top_lu['n_cases']):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'n={n}', va='center', fontsize=9, color='gray')
            ax.set_xlabel('Protest Rate (%)')
            ax.set_title('Protest Petition Rate by Proposed Land Use (≥5 cases)',
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "fig3_resistance_by_type.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  → Saved fig3_resistance_by_type.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SPATIAL / COUNCIL DISTRICT — "Where was this used? Council composition"
# ═══════════════════════════════════════════════════════════════════════════
def section4_spatial_council():
    print("\n=== 4. SPATIAL & COUNCIL DISTRICT ANALYSIS ===")
    zd = pd.read_csv(ENRICHED_ZD)
    pet = pd.read_csv(PET_SUMMARY)
    protested_cases = set(pet['case_number'].dropna().unique())
    zd['protested'] = zd['case_number'].isin(protested_cases).astype(int)

    # Parse year
    for col in ['application_start_date', 'status_date']:
        if col in zd.columns:
            zd[col] = pd.to_datetime(zd[col], errors='coerce')
    if 'application_start_date' in zd.columns:
        zd['app_year'] = zd['application_start_date'].dt.year

    # By council district
    if 'council_district' in zd.columns:
        # Note: Austin switched from at-large to 10 single-member districts in 2014
        cd_table = zd.groupby('council_district').agg(
            n_cases     = ('case_number', 'count'),
            n_protested = ('protested', 'sum'),
            protest_rate= ('protested', 'mean'),
        ).sort_values('n_cases', ascending=False)
        cd_table['protest_rate'] = (cd_table['protest_rate'] * 100).round(1)
        cd_table.to_csv(os.path.join(OUT_DIR, "protest_by_council_district.csv"))
        print("\n  By Council District:")
        print(cd_table.to_string())

    # ── Figure 4: Council district heatmap ──
    if 'council_district' in zd.columns and 'app_year' in zd.columns:
        pivot = zd[zd['protested']==1].groupby(['council_district', 'app_year']).size().unstack(fill_value=0)
        if pivot.shape[0] > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f'District {d}' for d in pivot.index])
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(int(y)) for y in pivot.columns], rotation=45, ha='right')
            # Annotate cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if val > 0:
                        ax.text(j, i, str(int(val)), ha='center', va='center',
                               fontsize=9, color='black' if val < pivot.values.max()*0.6 else 'white')
            plt.colorbar(im, ax=ax, label='# Protest Petitions')
            ax.set_title('Protest Petitions by Council District and Year',
                        fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "fig4_council_district_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  → Saved fig4_council_district_heatmap.png")

    # ── Spatial coordinates for protested cases (for timeline GIF later) ──
    protested_zd = zd[zd['protested']==1][['case_number', 'latitude', 'longitude', 'app_year',
                                            'proposed_land_use', 'council_district']].copy()
    protested_zd = protested_zd.dropna(subset=['latitude', 'longitude'])
    protested_zd.to_csv(os.path.join(OUT_DIR, "protested_cases_spatial.csv"), index=False)
    print(f"\n  → Saved protested_cases_spatial.csv ({len(protested_zd)} geolocated cases)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PROXIMITY / CONTAGION — "Is NIMBYism nearby? Construction nuisance?"
# ═══════════════════════════════════════════════════════════════════════════
def section5_proximity():
    print("\n=== 5. PROXIMITY & CONTAGION ANALYSIS ===")
    # Use panel v3 for protest_signed and zoning_case_nearby flags
    # Panel v3 is 1.8GB — read only the columns we need
    cols_needed = ['standardized_tcad_id', 'year', 'protest_signed',
                   'zoning_case_on_parcel', 'zoning_case_nearby',
                   'protest_nearby_area_pct', 'council_district',
                   'latitude', 'longitude']
    print("  Loading Panel V3 (selected columns)...")
    try:
        panel = pd.read_csv(PANEL_V3, usecols=cols_needed)
        print(f"  Panel shape: {panel.shape}")

        # Protest rates for parcels near vs far from zoning cases
        nearby = panel[panel['zoning_case_nearby']==1]
        not_nearby = panel[panel['zoning_case_nearby']==0]
        print(f"\n  Parcels near a zoning case:     {len(nearby):>10,} ({nearby['protest_signed'].mean()*100:.3f}% protest rate)")
        print(f"  Parcels NOT near a zoning case: {len(not_nearby):>10,} ({not_nearby['protest_signed'].mean()*100:.3f}% protest rate)")

        # Year-over-year protest signing rates
        yearly_protest = panel.groupby('year').agg(
            total_parcels  = ('standardized_tcad_id', 'count'),
            protestors     = ('protest_signed', 'sum'),
            protest_rate   = ('protest_signed', 'mean'),
            pct_near_zc    = ('zoning_case_nearby', 'mean'),
        ).reset_index()
        yearly_protest['protest_rate'] = (yearly_protest['protest_rate'] * 100).round(4)
        yearly_protest['pct_near_zc'] = (yearly_protest['pct_near_zc'] * 100).round(2)
        print("\n  Year-over-year panel summary:")
        print(yearly_protest.to_string(index=False))
        yearly_protest.to_csv(os.path.join(OUT_DIR, "panel_yearly_protest_rates.csv"), index=False)
        print(f"  → Saved panel_yearly_protest_rates.csv")

    except Exception as e:
        print(f"  ⚠ Could not load Panel V3: {e}")
        print("  Falling back to petition signers for proximity analysis...")

    # Petition signers distance-based analysis (always runs)
    pet_s = pd.read_csv(PET_SIGNERS)
    print(f"\n  Petition signers: {len(pet_s)} rows")
    print(f"  Cases spanned: {pet_s['case_number'].nunique()}")
    print(f"  Mean area_sqft per signer: {pet_s['area_sqft'].mean():.0f}")
    print(f"  Median area_pct per signer: {pet_s['area_pct'].median():.1f}%")

    # Distribution of signer area_pct (how close are signers to the project?)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pet_s['area_pct'].dropna(), bins=50, color=COLORS[0], alpha=0.85, edgecolor='white')
    ax.axvline(x=pet_s['area_pct'].median(), color=COLORS[1], linestyle='--',
               label=f'Median: {pet_s["area_pct"].median():.1f}%')
    ax.set_xlabel('Signer Area Percentage (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Protest Signer Area Shares',
                fontsize=14, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_signer_area_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved fig5_signer_area_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPREHENSIVE SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
def section6_summary():
    print("\n=== 6. COMPREHENSIVE SUMMARY TABLE ===")
    pet = pd.read_csv(PET_SUMMARY)
    pet_s = pd.read_csv(PET_SIGNERS)
    zd = pd.read_csv(ENRICHED_ZD)

    protested_cases = set(pet['case_number'].dropna().unique())
    zd['protested'] = zd['case_number'].isin(protested_cases).astype(int)

    summary_rows = [
        ("Total Protest Petition Cases", len(pet)),
        ("Year Range", f"{int(pet['year'].min())}-{int(pet['year'].max())}"),
        ("Total Parcels In Petition Dataset", len(pet_s)),
        ("Unique Parcels That Signed", pet_s[pet_s['signed']==1]['tcad_id'].nunique() if 'signed' in pet_s.columns else 'N/A'),
        ("Mean Signers Per Case", f"{pet['signers'].mean():.1f}"),
        ("Median Signer Percentage", f"{pet['signer_pct'].median():.1f}%"),
        ("Peak Year (Most Cases)", f"{int(pet.groupby('year').size().idxmax())} ({pet.groupby('year').size().max()} cases)"),
        ("Total Zoning Cases (enriched dataset)", len(zd)),
        ("  — Of which protested", zd['protested'].sum()),
        ("  — Protest rate across all zoning cases", f"{zd['protested'].mean()*100:.1f}%"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])
    summary_df.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    print(summary_df.to_string(index=False))
    print(f"  → Saved summary_table.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SPATIAL TIMELINE DATA — "NIMBYism spreading as a wildfire" (prep for GIF)
# ═══════════════════════════════════════════════════════════════════════════
def section7_spatial_timeline():
    print("\n=== 7. SPATIAL TIMELINE DATA (for GIF creation) ===")
    zd = pd.read_csv(ENRICHED_ZD)
    pet = pd.read_csv(PET_SUMMARY)

    # Merge protest data with spatial data
    pet_geo = pet.merge(
        zd[['case_number', 'latitude', 'longitude', 'proposed_land_use',
            'council_district', 'application_start_date']].drop_duplicates('case_number'),
        on='case_number', how='left'
    )
    pet_geo = pet_geo.dropna(subset=['latitude', 'longitude'])

    # Save for GIF creation
    pet_geo.to_csv(os.path.join(OUT_DIR, "protest_timeline_geo.csv"), index=False)
    print(f"  → Saved protest_timeline_geo.csv ({len(pet_geo)} geolocated protests)")
    print(f"  Geolocated: {len(pet_geo)}/{len(pet)} cases ({100*len(pet_geo)/len(pet):.0f}%)")

    # ── Figure 7: Static scatter by year ──
    if len(pet_geo) > 0:
        years = sorted(pet_geo['year'].dropna().astype(int).unique())
        n_years = len(years)
        ncols = min(6, n_years)
        nrows = (n_years + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3.5))
        if nrows == 1: axes = axes.reshape(1, -1)

        for idx, yr in enumerate(years):
            r, c = idx // ncols, idx % ncols
            ax = axes[r, c]
            sub = pet_geo[pet_geo['year'].astype(int) == yr]
            # Plot all previous years faintly
            prev = pet_geo[pet_geo['year'].astype(int) < yr]
            if len(prev) > 0:
                ax.scatter(prev['longitude'], prev['latitude'],
                          c='lightgray', s=10, alpha=0.3, zorder=1)
            # Plot current year
            ax.scatter(sub['longitude'], sub['latitude'],
                      c=COLORS[1], s=30, alpha=0.8, edgecolors='darkred',
                      linewidth=0.5, zorder=2)
            ax.set_title(f'{yr} (n={len(sub)})', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(n_years, nrows*ncols):
            r, c = idx // ncols, idx % ncols
            axes[r, c].set_visible(False)

        fig.suptitle('Spatial Spread of Zoning Protest Petitions in Austin (2007–2024)',
                    fontsize=15, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig7_spatial_spread.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Saved fig7_spatial_spread.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("AUSTIN ZONING PROTEST: DESCRIPTIVE STATISTICS")
    print("=" * 70)

    s1 = section1_petition_trends()
    section2_protest_vs_development()
    section3_zoning_type()
    section4_spatial_council()
    section5_proximity()
    section6_summary()
    section7_spatial_timeline()

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 70)
