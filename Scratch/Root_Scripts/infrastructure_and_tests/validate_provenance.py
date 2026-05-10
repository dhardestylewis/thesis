import pandas as pd
import numpy as np

print("Validating biweekly_panel.csv...")
bw_old = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv.backup', low_memory=False)
bw_new = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv', low_memory=False)

cols_to_sort = ['case_number', 'period_start']
bw_old = bw_old.sort_values(cols_to_sort).reset_index(drop=True)
bw_new = bw_new.sort_values(cols_to_sort).reset_index(drop=True)

common_cols = sorted(list(set(bw_old.columns).intersection(set(bw_new.columns))))

try:
    pd.testing.assert_frame_equal(bw_old[common_cols], bw_new[common_cols], check_dtype=False)
    print("SUCCESS: biweekly_panel.csv exactly matches the backup!")
except AssertionError as e:
    print("FAILED: biweekly_panel.csv does not match the backup!")
    print(e)

print("\nValidating annualized_all_parcel_panel.parquet...")
ann_old = pd.read_parquet(r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML\annualized_all_parcel_panel.parquet.backup')
ann_new = pd.read_parquet(r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML\annualized_all_parcel_panel.parquet')

cols_to_sort = ['parcel_id_10', 'year']
ann_old = ann_old.sort_values(cols_to_sort).reset_index(drop=True)
ann_new = ann_new.sort_values(cols_to_sort).reset_index(drop=True)

common_cols = sorted(list(set(ann_old.columns).intersection(set(ann_new.columns))))

try:
    pd.testing.assert_frame_equal(ann_old[common_cols], ann_new[common_cols], check_dtype=False)
    print("SUCCESS: annualized_all_parcel_panel.parquet exactly matches the backup!")
except AssertionError as e:
    print("FAILED: annualized_all_parcel_panel.parquet does not match the backup!")
    print(e)
