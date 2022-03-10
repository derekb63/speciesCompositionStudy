import mpltern
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors


sns.set(font_scale=2)

data = pd.read_csv('ignition_delay_values.csv')
data['LIG'] = data.filter(regex='LIG').sum(axis=1)
data['CELL'] = pd.to_numeric(data['CELL'], errors='coerce')
fig = plt.figure()
axs_2 = fig.add_subplot(111, projection='ternary')
pc = axs_2.scatter(data['CELL'].values,
                   data['HCELL'].values,
                   data['LIG'].values,
                   c=data['ignition_delay'].values, cmap='magma')
cax = axs_2.inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs_2.transAxes)
axs_2.set_tlabel('Cellulose')
axs_2.set_llabel('Hemi-Cellulose')
axs_2.set_rlabel('Lignin')
axs_2.taxis.set_label_position('tick1')
axs_2.laxis.set_label_position('tick1')
axs_2.raxis.set_label_position('tick1')
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label(r'$\tau$ (s)', rotation=270, va='baseline')
plt.show()

fig = plt.figure()
axs_2 = fig.add_subplot(111, projection='ternary')
pc = axs_2.scatter(data['CELL'].values,
                   data['HCELL'].values,
                   data['LIG'].values,
                   norm=colors.LogNorm(),
                   c=0.006/data['transit_distance'].values, cmap='magma')
cax = axs_2.inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs_2.transAxes)
axs_2.set_tlabel('Cellulose')
axs_2.set_llabel('Hemi-Cellulose')
axs_2.set_rlabel('Lignin')
axs_2.taxis.set_label_position('tick1')
axs_2.laxis.set_label_position('tick1')
axs_2.raxis.set_label_position('tick1')
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label('Da', rotation=270, va='baseline')
plt.show()
