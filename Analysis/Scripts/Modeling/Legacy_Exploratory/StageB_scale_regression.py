import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    df = pd.read_csv(r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv', on_bad_lines='skip', engine='python')
    df = df.dropna(subset=['delta_max_far', 'gross_site_area_acres', 'year'])
    X = df[['gross_site_area_acres', 'year']]
    y = df['delta_max_far']

    cb = CatBoostRegressor(iterations=50, depth=4, learning_rate=0.05, verbose=0)
    cb.fit(X, y)
    p = cb.predict(X)
    print(f'MAE: {mean_absolute_error(y, p):.4f}')
    print(f'R2: {r2_score(y, p):.4f}')
except Exception as e:
    print("ERROR:", e)
