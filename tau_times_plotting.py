import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

plt.rcParams.update({'font.size': 22})
tau_values = pd.read_csv('tau_values.csv')
alpha_values = pd.Series({'Douglas Fir Bark': 5.753943740011115e-07,
                'Douglas Fir Wood': 2.851289391405382e-07,
                'Oak Wood': 1.1923060561250857e-07,
                'Pine Wood': 2.431337661615995e-07,
                'Wheat Straw': 3.313768321792447e-07})
normalized_values = tau_values*alpha_values
ax = sns.lineplot(data=normalized_values, linewidth=3)
# ax = sns.lineplot(data=tau_values.set_index('Time'), linewidth=3)
ax.set(xlabel='Time (s)', ylabel=r'$\tau \; (s)$')
plt.tight_layout()
plt.show()
