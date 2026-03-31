import sys
sys.path.append(r'c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Modeling')
import StageC_opposition_risk as sc

path = r'c:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv'
print("Running single horizon evaluation...")
sc.process_horizon(path, 'H0_Only_Complete')
