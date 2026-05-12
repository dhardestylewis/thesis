import geopandas as gpd
import matplotlib.pyplot as plt

try:
    print("Loading pre-processed zoning case environments...")
    path = r'c:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\GeoJSON\zoning_cases_with_nearby_parcels.geojson'
    
    gdf = gpd.read_file(path)
    
    # Filter for our specific case C14-2007-0131
    case_gdf = gdf[gdf['Case Number'].astype(str).str.contains('C14-2007-0131', na=False)]
    
    if case_gdf.empty:
        print("Couldn't find case C14-2007-0131 in this dataset!")
    else:
        print(f"Found {len(case_gdf)} parcels associated with C14-2007-0131")
        
        # The 14 protesting petition IDs explicitly provided by the user in the prompt / raw notes earlier
        target_ids = ['0209060502', '0209060503', '0209060504', '0209060506', '0209060507', 
                      '0209060508', '0209060509', '0209060606', '0209060607', '0209060901', 
                      '0209060911', '0209061001', '0209061011', '0209061013']
                      
        # Clean the parcel ID column to ensure matching
        case_gdf.loc[:, 'clean_id'] = case_gdf['nearby_parcel_id_10'].astype(str).str.replace('-', '').str.replace('.0', '', regex=False)
        protesters = case_gdf[case_gdf['clean_id'].isin(target_ids)]
        non_protesters = case_gdf[~case_gdf['clean_id'].isin(target_ids)]
            
        print(f"Found {len(protesters)} explicitly protesting parcels")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot non-protesting nearby parcels (within 200ft)
        if not non_protesters.empty:
            non_protesters.plot(ax=ax, facecolor='#EEEEEE', edgecolor='gray', alpha=0.7, label='Neighborhood Parcels')
            
        # Plot protesting parcels
        if not protesters.empty:
            protesters.plot(ax=ax, facecolor='#99CCFF', edgecolor='blue', linewidth=1.5, hatch='//', label='Valid Protesting Parcel')
            
        import matplotlib.patches as mpatches
        handles = []
        if not protesters.empty:
            handles.append(mpatches.Patch(facecolor='#99CCFF', edgecolor='blue', hatch='//', label='Valid Protesting Parcel'))
        handles.append(mpatches.Patch(facecolor='#EEEEEE', edgecolor='gray', label='Non-Protesting Parcel'))
        
        plt.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(handles), parse_math=False, fontsize=10)
        
        plt.title("Visualizing the statutory 200ft Protest Buffer\n(Empirical TCAD Parcel Boundaries for Case C14-2007-0131)", fontsize=14, fontweight='bold')
        ax.set_axis_off()
        plt.tight_layout()
        
        out_path = r'c:\Users\dhl\data\thesis\thesis\Figures\ch2\fig_ch2_01_waller_buffer_map.png'
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Map updated and saved to {out_path}")

except Exception as e:
    import traceback
    traceback.print_exc()
