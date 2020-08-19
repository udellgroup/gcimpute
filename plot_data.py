import pandas as pd
import matplotlib.pyplot as plt

online_data = pd.read_csv('averaged_data.csv')
offline_data = {'Continuous' : 1.0192609127294214,
                'Ordinal' : 1.0218395758637964,
                'Binary' : 0.9679910340208437}

print(online_data['Continuous'])
print(len(online_data['Continuous']))
plt.xlabel('Batch number')
plt.ylabel('SMAE')
plt.plot(range(120), online_data['Continuous'], 'bo')
plt.plot(range(120), [offline_data['Continuous']] * 120, 'ro')
plt.show()