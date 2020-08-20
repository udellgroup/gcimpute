import pandas as pd
import matplotlib.pyplot as plt

online_data = pd.read_csv('averaged_data.csv')
offline_data = {'Continuous' : 1.0192609127294214,
                'Ordinal' : 1.0218395758637964,
                'Binary' : 0.9679910340208437}
offline_data = pd.read_csv('averaged_stdata.csv')

type = 'Binary'
small_type = 'bin'

print(online_data[type])
print(len(online_data[type]))
plt.xlabel('Batch number')
plt.ylabel('SMAE')
plt.plot(range(120), online_data[type], 'bo', label='Online ('+small_type+')')
plt.plot(range(120), offline_data[type], 'ro', label='Offline ('+small_type+')') # [offline_data[type]] * 120
plt.legend()
plt.show()