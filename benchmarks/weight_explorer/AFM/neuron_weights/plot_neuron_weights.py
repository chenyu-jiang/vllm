import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('l0_e0.csv')

weights = df["importance"].sort_values().reset_index(drop=True)
print(weights)
# plot distrobution
fig, ax = plt.subplots()
sns.lineplot(weights, ax=ax)
ax.set_title('Sorted Neuron Importance')
ax.set_xlabel('Neuron Idx (Sorted)')
ax.set_ylabel('Normalized Importance')
fig.savefig('neuron_importance_l0_e0.pdf', bbox_inches='tight')