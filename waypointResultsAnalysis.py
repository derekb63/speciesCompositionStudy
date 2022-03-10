import numpy as np
import pandas as pd
import mpltern
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
plt.rcParams.update({'font.size': 18})

if __name__ == "__main__":
    results = pd.DataFrame(data={"Douglas-fir Bark": [2.40, 97.6, 84.7, 68.2, 0.5, 52.8],
                                 "Douglas-fir": [2.90, 97.1, 75.5, 88.8, 0.1, 28.9],
                                 "Wheat Straw": [6.20, 93.8, 52.3, 72.7, 8.1, 6.47],
                                 "Oak": [0.20, 99.8, 75.5, 90.8, 0.5, 19.4],
                                 "Pine": [2.10, 97.9, 76.2, 89.9, 0.3, 28.8]},
                           index=['moisture', 'dry_matter', 'ADF', 'NDF', 'ash', 'LIG'])

    results = results.transpose()
    results['CELL'] = results['ADF'] - results['LIG']
    results['HCELL'] = results['NDF'] - results['ADF']

    results[['CELLn', 'HCELLn', 'LIGn']] = results[['CELL', 'HCELL', 'LIG']].div(
        results[['CELL', 'HCELL', 'LIG']].sum(axis=1), axis=0)

    database_values = pd.read_csv('database_concentration.csv', index_col=0)
    results['P50'] = database_values.loc[results.index]['P50']

    fig, axs = plt.subplots(nrows=1, ncols=1)
    width = 16
    fig.set_figwidth(width)
    fig.set_figheight(width * 3 / 4)
    axs.axis('off')
    axs_2 = fig.add_subplot(111, projection='ternary')

    pc = axs_2.scatter(database_values['CELLn'],
                       database_values['HCELLn'],
                       database_values['LIGn'],
                       c=database_values['P50'], cmap='copper',
                       s=156, label='Database')
    pc2 = axs_2.scatter(results['CELLn'],
                        results['HCELLn'],
                        results['LIGn'],
                        marker='s',
                        c=results['P50'], cmap='copper',
                        s=156, label='Tested')

    for name, row in results.iterrows():
        origin = [database_values.loc[name]['CELLn'],
                  database_values.loc[name]['HCELLn'],
                  database_values.loc[name]['LIGn']]
        destination = [row['CELLn'], row['HCELLn'], row['LIGn']]
        quiver_lines = axs_2.quiver(*origin, *destination)
        quiver_lines.scale = 1
        quiver_lines.units = 'xy'
        quiver_lines.width = 0.005
        quiver_lines.color = 'r'
    scatter_text = []
    for name, row in database_values.iterrows():
        scatter_text.append(axs_2.text(row['CELLn'], row['HCELLn'], row['LIGn'], name,
                                       horizontalalignment='center', verticalalignment='top',
                                       path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")]
                                       )
                            )

    #     scatter_text.append(axs_2.text(row['CELLn'], row['HCELLn'], row['LIGn'], name,
    #                                    horizontalalignment='center', verticalalignment='top'))
    adjust_text(scatter_text)
    # adjust_text(scatter_text, arrowprops=dict(arrowstyle='->', color='black'))
    # database_values.apply(lambda i: axs_2.text(i['CELL'],
    #                  i['HCELL'],
    #                  i['LIG'],
    #                  i['SAMPLE']+''+i['ORGAN FRACTION'],
    #                  horizontalalignment='center', verticalalignment='top',fontsize=12), axis=1)
    cax = axs_2.inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs_2.transAxes)
    axs_2.set_tlabel('Cellulose')
    axs_2.set_llabel('Hemi-Cellulose')
    axs_2.set_rlabel('Lignin')
    axs_2.taxis.set_label_position('tick1')
    axs_2.laxis.set_label_position('tick1')
    axs_2.raxis.set_label_position('tick1')
    axs_2.legend()
    colorbar = fig.colorbar(pc, cax=cax)
    colorbar.set_label('P$_{50}$ ($^{\circ}$C)', rotation=270, va='baseline')
    plt.show()
    plt.tight_layout()