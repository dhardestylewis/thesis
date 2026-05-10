import pandas as pd
df = pd.DataFrame({'year': [2018, 2019, 2021], 'geo': ['A', 'A', 'A'], 'val': [1, 2, 3]})
df = df.sort_values('year')
df['shift_target'] = df['val'].shift(-1)

# Proper target
target_dict = df.set_index(['geo', 'year'])['val'].to_dict()
df['proper_target_1yr'] = df.apply(lambda r: target_dict.get((r['geo'], r['year'] + 1)), axis=1)

print(df)
