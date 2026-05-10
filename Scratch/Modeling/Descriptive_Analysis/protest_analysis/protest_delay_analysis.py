import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv', low_memory=False)
df['period_start'] = pd.to_datetime(df['period_start'], errors='coerce')

# T0 per case = first period_start
t0 = df.groupby('case_number')['period_start'].min().rename('t0')

# Petition events only
pet = df[df['petition_event'] == 1][['case_number','period_start','period_seq']].copy()
pet = pet.merge(t0, on='case_number')
pet['days_to_petition'] = (pet['period_start'] - pet['t0']).dt.days
pet['months_to_petition'] = pet['days_to_petition'] / 30.4

# One row per case (first petition if multiple)
first_pet = pet.sort_values('period_start').groupby('case_number').first()

print('=== Days from T0 to petition event ===')
print(first_pet['days_to_petition'].describe().round(1))
print()
print('=== Months from T0 to petition event ===')
print(first_pet['months_to_petition'].describe().round(1))
print()
med = first_pet['months_to_petition'].median()
p75 = first_pet['months_to_petition'].quantile(0.75)
p90 = first_pet['months_to_petition'].quantile(0.90)
print(f'Median:   {med:.1f} months')
print(f'75th pct: {p75:.1f} months')
print(f'90th pct: {p90:.1f} months')

# Period_seq at petition — how far into the lifecycle
print()
print('=== Period_seq at petition (biweekly steps into lifecycle) ===')
print(first_pet['period_seq'].describe().round(1))

# Total case duration by protest status
case_end = df.groupby('case_number')['period_start'].max().rename('t_end')
case_dur = t0.to_frame().join(case_end)
case_dur['total_months'] = (case_dur['t_end'] - case_dur['t0']).dt.days / 30.4
case_dur['protested'] = case_dur.index.isin(first_pet.index)

print()
print('=== Total case duration (months) by protest status ===')
print(case_dur.groupby('protested')['total_months'].describe().round(1))

extra = case_dur.groupby('protested')['total_months'].median()
print()
print(f'Median duration NON-protested: {extra[False]:.1f} months')
print(f'Median duration PROTESTED:     {extra[True]:.1f} months')
print(f'Implied delay from protest:    {extra[True] - extra[False]:.1f} months')
