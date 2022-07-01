import os
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn
from adjustText import adjust_text
import mpltern
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
import plotly.graph_objects as go

plt.rcParams.update({'font.size': 22})


def fit_model(x_values, y_values):
    x_values = x_values.reshape(-1, 1)
    y_values = y_values.reshape(-1, 1)
    clf = LogisticRegression(random_state=0)
    clf.fit(x_values, y_values)
    return clf


def find_outputs(directory_paths):
    filenames = []
    for directory in directory_paths:
        for folder, _, files in os.walk(directory):
            for filename in files:
                if 'output' in filename.lower() and '.json' in filename.lower():
                    filenames.append(os.path.join(directory, os.path.join(folder, filename)))
    return filenames


def load_data(directory_list, remove_some_uneeded_values=True):
    output_files = [x for x in find_outputs(directory_list) if 'test' in x.lower()]
    frame_list = []
    for f in output_files:
        test_frame = pd.read_json(f)
        test_frame = test_frame.bfill(axis=1)['fuel_morphology']
        test_frame.rename('_'.join(f.split('/')[-1].split('_')[:2]), inplace=True)
        frame_list.append(test_frame)
    file_data = pd.concat(frame_list, axis=1).transpose()
    if remove_some_uneeded_values == True:
        file_data.drop(['moisture_content', 'particle_size', 'typical_morphology',
                        'container_volume', 'sample_mass', 'notes', 'apparatus'], axis=1, inplace=True)
    return file_data


def get_probability(model, probability_target=0.5):
    diff = []
    for i in range(850):
        output_value = model.predict_proba(np.array(i).reshape(-1, 1))[0][1]
        diff.append(abs(output_value - probability_target))
    return diff.index(min(diff))


def load_concentration_data(species_data_file='/home/derek/Documents/IgnitionProject/ef5b01753_si_001.csv'):
    with open(species_data_file, 'r') as f:
        concentration_frame = pd.read_csv(f)
    concentration_frame = concentration_frame[[x for x in concentration_frame.columns if 'unnamed' not in x.lower()]]
    concentration_frame.drop(
        columns=['TAXONOMY', 'REF.', 'beta', 'gamma', 'delta', 'epsion', 'alpha', 'VARIETY', 'N', 'S', 'Cl', 'Ash'],
        inplace=True)
    concentration_frame.rename(columns={concentration_frame.columns[0]: 'CATEGORY'}, inplace=True)
    concentration_frame.replace('Outsider', None, inplace=True)
    concentration_frame['CELL'] = concentration_frame['CELL'].astype(float)
    return concentration_frame


line_style_dict = {0.1: ['dotted',  seaborn.color_palette('tab10')[0]],
                   5.8: ['dashdot', seaborn.color_palette('tab10')[2]]}

def find_matches_composition(frame, column, text):
    rows = frame[frame[column[0]].str.match(text[0], na=False)]
    rows = rows[rows[column[1]].str.match(text[1], na=False)]
    rows['LIG'] = rows.filter(regex='LIG').sum(axis=1)
    return rows[['SAMPLE', 'ORGAN FRACTION', 'CELL', 'HCELL', 'LIG']]


def fuel_bed_flux(test_dataframe):
    heater_total_area = np.pi*0.00635*0.051 # area  (m**2)
    heat_loss_fit_functions = {5.8: np.array([-9.80236242e-08, 2.58938517e-04, -2.40139019e-01, 1.07592555e+02,
                                               -1.65462363e+04]),
                               3.54: np.array([-9.11357090e-08, 2.43777951e-04, -2.29606324e-01, 1.02832394e+02,
                                               -1.60117540e+04]),
                               0.51: np.array([-7.53661929e-08, 2.09068638e-04, -2.05492279e-01, 9.19342571e+01,
                                               -1.47880853e+04]),
                               0.10: np.array([-7.53661929e-08, 2.09068638e-04, -2.05492279e-01, 9.19342571e+01,
                                               -1.47880853e+04])
                               }
    heat_loss_coeffs = test_dataframe['wind_speed'].apply(lambda x: heat_loss_fit_functions[x])
    heat_loss_flux = pd.Series(index=test_dataframe.index, dtype=np.float64)
    for idx, val in test_dataframe['mean_temperature'].items():
        heat_loss_flux[idx] = np.polyval(heat_loss_coeffs[idx], val)
    bed_power = test_dataframe['mean_power'] - (heat_loss_flux * (heater_total_area/2))
    bed_flux = bed_power / (heater_total_area/2)
    return bed_flux


model_results = {'Douglas-fir': 574, 'Pine': 634, 'Douglas-fir Bark': 800, 'Oak': 663, 'Wheat Straw': 800, 'Pine Bark': 800}

if __name__ == '__main__':
    sns.set(font_scale=2.5)
    concentration_data = load_concentration_data()
    data_file = '/home/derek/Documents/IgnitionData/SpeciesData/species_data.csv'
    directories = ['/home/derek/Documents/IgnitionData/SpeciesData/']
    test_data = load_data(directories)
    test_data['bed_flux'] = fuel_bed_flux(test_data)
    data = pd.read_csv(data_file)
    data['ignition'] = data['ignition'].astype(bool)
    data.groupby('wind_speed')
    material_names = ['Douglas-fir', 'Pine', 'Douglas-fir Bark', 'Oak', 'Wheat Straw', 'Pine Bark']
    composition_names = [['Douglas', 'Untreated Wood'], ['Pine', 'Untreated Wood'], ['Douglas', 'Bark'],
                         ['Oak', 'Untreated Wood'], ['Wheat', 'Straw'], ['Pine', 'Bark']]
    composition_values = pd.DataFrame(index=material_names, columns=['SAMPLE', 'ORGAN FRACTION', 'CELL', 'HCELL', 'LIG'])
    test_result_data = pd.DataFrame(index=material_names, columns=['temperature', 'ignition'])
    for idx, val in enumerate(material_names):
        comp_temp = find_matches_composition(concentration_data, ['SAMPLE', 'ORGAN FRACTION'], composition_names[idx])
        print(comp_temp)
        comp_temp['CELL'] = comp_temp['CELL'].astype(float)
        comp_temp['ORGAN FRACTION'] = comp_temp['ORGAN FRACTION'].value_counts().idxmax()
        comp_temp['SAMPLE'] = comp_temp['SAMPLE'].value_counts().idxmax()
        comp_temp[['CELL', 'HCELL', 'LIG']] = comp_temp.mean()
        composition_values.loc[val] = comp_temp.iloc[0]
    #
    # model_results = {}
    # for material_type in data.groupby(['material', 'wind_speed']):
    #     try:
    #         temp_model = fit_model(material_type[1]['temperature'].values, material_type[1]['ignition'].values)
    #         model_results[material_type[0]] = get_probability(temp_model)
    #     except ValueError:
    #         pass
    # model_results.update({('Douglas-fir Bark', 0.1): 800,
    #                       ('Wheat Straw', 0.1): 800,
    #                       ('Douglas-fir Bark', 5.8): 800,
    #                       ('Pine Bark', 5.8): 800})
    #
    # # estimate the heat flux boundary condition at the p_50 temperature
    # flux_fit_coeffs = np.polyfit(test_data['mean_temperature'].astype(np.float),
    #                              test_data['bed_flux'].astype(np.float),
    #                              deg=2)
    # flux_bc_values = {key: int(np.polyval(flux_fit_coeffs, model_results[key])) for key in model_results}
    # x_column_name = 'temperature'
    # fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    # axs_ytwin = [ax.twinx() for ax in axs]
    # width = 8
    # fig.set_figwidth(width)
    # fig.set_figheight(width)
    # marker_s = 75
    # for num, item in enumerate(data.groupby('wind_speed')):
    #     for temp_data in item[1].groupby(['material']):
    #         wind_speed = temp_data[1]['wind_speed'].unique()[0]
    #         material = temp_data[1]['material'].unique()[0]
    #         # plot_data = temp_data[1].reset_index()
    #         plot_data = temp_data[1]
    #         plot_data = plot_data[['ignition', x_column_name]].astype('float')
    #         series_label = '{0:.1f} (m/s),{1} '.format(wind_speed, material)
    #         series_color = line_style_dict[wind_speed][1]
    #         axs[num].grid(visible=True, linestyle='dotted', color='lightgray')
    #         a = sns.regplot(x=x_column_name,
    #                         y='ignition',
    #                         data=plot_data,
    #                         logistic=True,
    #                         ax=axs[num],
    #                         label=series_label,
    #                         # scatter_kws={'s': marker_s, 'color': series_color},
    #                         # line_kws={'linestyle': line_style_dict[wind_speed][0], 'color': series_color},
    #                         ci=95)
    #         axs_ytwin[num].set_ylabel(None)
    #         axs[num].set_xlabel(None)
    #         axs[num].set_ylabel(None)
    #         # axs[num].title.set_text('{0:.0f}$^\circ$'.format(heater_lead_angle))
    # axs[1].set_ylabel('Ignition Probability')
    # axs_ytwin[1].set_ylabel('Test Outcome')
    # [ax.set_yticks([0.0, 1.0]) for ax in axs_ytwin]
    # [ax.set_yticks([1.0, 0.5, 0.0]) for ax in axs]
    # [x.set_yticklabels(["No Ignition", "Ignition"]) for x in axs_ytwin]
    # axs[-1].set_xlabel(r'Heater Temperature($^\circ$C)')
    # # axs[-1].set_xlabel(x_column_name.title().replace('_', " ")+'($^\circ$C)')
    # handles, labels = axs[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc=10, bbox_to_anchor=(0.75, 0.825), facecolor='white', framealpha=1)
    # # fig.suptitle(heater_lead_angle)
    # fig.tight_layout()
    ternary_data = composition_values
    ternary_data['P50'] = pd.DataFrame.from_dict(model_results, orient='index')

    ternary_data[['CELLn', 'HCELLn', 'LIGn']] = ternary_data[['CELL', 'HCELL', 'LIG']].div(ternary_data[['CELL', 'HCELL', 'LIG']].sum(axis=1), axis=0)

    fig, axs = plt.plot(projection='ternary')
    width = 28
    fig.set_figwidth(width)
    fig.set_figheight(width * 9/21)
    marker_s = 128
    # axs[1].axis('off')
    annotate_list = []
    # for temp_data in data.groupby(['material', 'wind_speed']):
    #     sns.regplot(x='temperature', y='ignition', data=temp_data[1], logistic=True, ax=axs, label=temp_data[0],
    #                 scatter_kws={'s': marker_s}, ci=95)
        # axs[0].scatter(model_results[temp_data[0]], 0.5, label=temp_data[0] + ' P(0.5)', marker='x', s=marker_s)
        # annotate_list.append(axs[0].annotate(model_results[temp_data[0]], (model_results[temp_data[0]], 0.5),
        #                      horizontalalignment='left', verticalalignment='top'))
    # adjust_text(annotate_list)
        # sns.regplot(x='temperature', y='ignition', data=pine, logistic=True, ax=axs[0], label='Pine')
    # axs[0].scatter(df_50, 0.5, label='Douglas-fir P(0.5)')
    # axs[0].scatter(pine_50, 0.5, label='Pine P(0.5)')
    # axs[0].annotate(df_50, (df_50, 0.5), horizontalalignment='left', verticalalignment='top')
    # axs[0].annotate(pine_50, (pine_50, 0.5), horizontalalignment='left', verticalalignment='top')
    # axs.set_ylabel('Test Outcome')
    # axs.set_yticks([1.0, 0.0])
    # axs.set_yticklabels(["Ignition", "No Ignition"])
    # axs.set_xlabel('Heater Setpoint ($^{\circ}$C)')
    # axs.legend()
    #
    # axs_2 = fig.add_subplot(111, projection='ternary')
    pc = axs.scatter(ternary_data['CELLn'],
                       ternary_data['HCELLn'],
                       ternary_data['LIGn'])
                       # c=ternary_data['P50'], cmap='copper', s=marker_s)

    scatter_text = []
    for row in ternary_data.iterrows():
        i = row[1]
        scatter_text.append(axs.text(i['CELL'], i['HCELL'], i['LIG'], row[0],
                                       horizontalalignment='center', verticalalignment='top'))
    adjust_text(scatter_text)
    # adjust_text(scatter_text, arrowprops=dict(arrowstyle='->', color='black'))
    # ternary_data.apply(lambda i: axs_2.text(i['CELL'],
    #                  i['HCELL'],
    #                  i['LIG'],
    #                  i['SAMPLE']+''+i['ORGAN FRACTION'],
    #                  horizontalalignment='center', verticalalignment='top',fontsize=12), axis=1)
    cax = axs.inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs.transAxes)
    axs.set_tlabel('Cellulose')
    axs.set_llabel('Hemi-Cellulose')
    axs.set_rlabel('Lignin')
    axs.taxis.set_label_position('tick1')
    axs.laxis.set_label_position('tick1')
    axs.raxis.set_label_position('tick1')
    colorbar = fig.colorbar(pc, cax=cax)
    colorbar.set_label('P$_{50}$ ($^{\circ}$C)', rotation=270, va='baseline')
    plt.show()
    plt.tight_layout()
    # #
    # # fig_test_2 = go.Figure(go.Scatterternary({
    # #     'mode': 'markers',
    # #     'a': [0.44, 0.40, 0.26, 0.45, 0.43],
    # #     'b': [0.22, 0.297, 0.11, 0.22, 0.27],
    # #     'c': [0.27, 0.20, 0.59, 0.24, 0.19],
    # #     'text': ['df', 'pine', 'dfb', 'oak', 'ws'],
    # #     'marker': {
    # #         'symbol': 100,
    # #         'color': '#DB7365',
    # #         'size': 14,
    # #         'line': {'width': 2}
    # #     }
    # # }))
    # # fig_test_2.write_html('fig_test_ternary_2.html')
