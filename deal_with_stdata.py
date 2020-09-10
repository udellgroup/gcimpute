import pandas as pd
import numpy as np

data = []

for i in range(10):
    trial = pd.read_csv('data_'+str(i)+'_std.csv')
    for type in ['Continuous', 'Ordinal', 'Binary']:
        for col in range(5):
            trial[type][col] = float(trial[type][col])
    data.append(trial)

avg_vals = {'Continuous' : np.empty_like(data[0]['Continuous'], float),
            'Ordinal' : np.empty_like(data[0]['Ordinal'], float),
            'Binary' : np.empty_like(data[0]['Binary'], float)}

for i in range(10):
    avg_vals['Continuous'] += np.array(data[i]['Continuous'])
    avg_vals['Ordinal'] += np.array(data[i]['Ordinal'])
    avg_vals['Binary'] += np.array(data[i]['Binary'])

avg_vals['Continuous'] /= 10
avg_vals['Ordinal'] /= 10
avg_vals['Binary'] /= 10

avg_vals['Continuous'] = sum(avg_vals['Continuous']) / 5
avg_vals['Ordinal'] = sum(avg_vals['Ordinal']) / 5
avg_vals['Binary'] = sum(avg_vals['Binary']) / 5

print(avg_vals)

# df = pd.DataFrame(avg_vals)
# df.to_csv('averaged_stdata.csv')