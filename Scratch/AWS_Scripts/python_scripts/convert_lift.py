import sys
import pandas as pd

try:
    df = pd.read_csv(sys.stdin)
    df['Lift'] = df['PR_AUC'] / df['Naive_PR_AUC']
    
    for year in sorted(df['Test_Year'].unique()):
        print(f'\n=== Walk-Forward Cutoff: {year} ===')
        year_df = df[df['Test_Year'] == year]
        
        for horizon in ['14_Days', '3_Months', '6_Months', '1_Year', '2_Years']:
            hdf = year_df[year_df['Horizon'] == horizon]
            if not hdf.empty:
                for _, row in hdf.iterrows():
                    print(f"  [{horizon:<10}] {row['Model']:<15} Lift: {row['Lift']:>6.2f}x (PR: {row['PR_AUC']:.4f} / Base: {row['Naive_PR_AUC']:.4f})")
except Exception as e:
    print(e)
