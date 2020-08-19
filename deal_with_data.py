import pandas as pd
import numpy as np

data = []

def parse_str(raw):
    return [float(n) for n in str.split(raw[1:-1])]

batch_number = 120
for i in range(10):
    trial = pd.read_csv('data_'+str(i)+'.csv')
    for type in ['Continuous', 'Ordinal', 'Binary']:
        for batch in range(batch_number):
            trial[type][batch] = parse_str(trial[type][batch])
    data.append(trial)

# batch_number = len(data[0]['Continuous'])
avg_vals = {'Continuous' : np.empty_like(data[0]['Continuous'], float),
            'Ordinal' : np.empty_like(data[0]['Ordinal'], float),
            'Binary' : np.empty_like(data[0]['Binary'], float)}

for i in range(10):
    avg_vals['Continuous'] += np.array([sum(cols) / 5 for cols in data[i]['Continuous']])
    avg_vals['Ordinal'] += np.array([sum(cols) / 5 for cols in data[i]['Ordinal']])
    avg_vals['Binary'] += np.array([sum(cols) / 5 for cols in data[i]['Binary']])

avg_vals['Continuous'] /= 10
avg_vals['Ordinal'] /= 10
avg_vals['Binary'] /= 10

df = pd.DataFrame(avg_vals)
df.to_csv('averaged_data.csv')