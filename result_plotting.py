import os
import json
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

def find_inputs(directory_paths):
    '''
        Find all of the result files for the ignition tests

        Inputs:
        - directory_paths: the directory to look for result files (list)

        Outputs:
        - filenames: list of complete paths to all of the files (list)
    '''
    filenames = []
    dir_counter = 0
    if type(directory_paths) is str:
        directory_paths = [directory_paths]
    for directory in directory_paths:
        for dirs in os.listdir(directory):
            # Comment this section out if the parent folders are more than one deep
            #             if os.path.isdir(dirs) == False:
            #                 dirs = ''
            for _, _, files in os.walk(os.path.join(directory, dirs)):
                for filename in files:
                    if ('test' in filename.lower() and '.json' in filename.lower()
                            and 'input' not in filename.lower()):
                        filenames.append(os.path.join(directory, dirs, filename))
    return filenames


def scrape_data(file_name, verbose=False):
    with open(file_name, 'r+', encoding='utf-8') as f:
        if verbose == True:
            print(file_name)
        data = json.load(f)
    try:
        relevant_data = {'time_to_ignition': data['results']['time_to_ignition'],
                         'total_energy': data['results']['total_energy'],
                         # 'power_values': data['results']['power_values'],
                         'setpoint': data['test_parameters']['setpoint'],
                         'density': data['test_parameters']['density'],
                         'moisture_content': data['fuel_morphology']['moisture_content'],
                         'species': data['fuel_morphology']['species'],
                         'ignition_temperature': data['results']['temperature_at_ignition'],
                         'date': data['test_parameters']['date'],
                         'particle_size': data['fuel_morphology']['particle_size'],
                         'sample_mass': data['test_parameters']['sample_mass'],
                         'apparatus': data['test_parameters']['apparatus'],
                         'lower_power': data['results']['lower_power'],
                         'mean_power': data['results']['mean_power']}
    except KeyError:
        print('File Error:', file_name)
        relevant_data = {'time_to_ignition': data['results']['time_to_ignition'],
                         'total_energy': data['results']['total_energy'],
                         'setpoint': data['test_parameters']['setpoint'],
                         'density': data['test_parameters']['density'],
                         'sample_mass': data['test_parameters']['sample_mass'],
                         'moisture_content': data['fuel_morphology']['moisture_content'],
                         'species': data['fuel_morphology']['species'],
                         'ignition_temperature': data['results']['temperature_at_ignition'],
                         'date': data['test_parameters']['date'],
                         'particle_size': data['fuel_morphology']['particle_size'],
                         'apparatus': data['test_parameters']['apparatus']}
    return relevant_data

if __name__ == "__main__":
    # Look for the output files and load the data in the files to a DataFrame
    data_directory = '/home/derek/Documents/IgnitionData/SpeciesData/'
    data_files = find_inputs(data_directory)
    test_results = pd.DataFrame(map(scrape_data, data_files))

    previous_douglas_fir = pd.read_pickle('/home/derek/Documents/IgnitionProject/pickle_files/ignition_data.pkl')
    previous_douglas_fir = previous_douglas_fir[previous_douglas_fir.particle_size == '#18 granulated']
    # add a column to state if ignition happened
    test_results['ignition'] = ~test_results['time_to_ignition'].isna()
    # Group the data based on the species and ignition
    grouped_results = test_results.groupby(['species', 'ignition'])
    plt.rcParams.update({'font.size': 18, 'lines.markersize': 14})
    color_dict = {v: sns.color_palette()[i] for i, v in enumerate(test_results.species.unique())}
    marker_dict = {v: Line2D.filled_markers[i] for i, v in enumerate(test_results.species.unique())}
    fig, ax = plt.subplots(2, 2, figsize=(18, 12), sharey=True, sharex=True)
    for label, df in grouped_results:
        color = color_dict[label[0]]
        marker = marker_dict[label[0]]
        if not label[-1]:
            sns.scatterplot(x="setpoint", y="mean_power", data=df, label=label[0], ax=ax[0][0], color=color, marker=marker)
            sns.scatterplot(x="setpoint", y="lower_power", data=df, label=label[0], ax=ax[1][0], color=color, marker=marker)
        else:
            sns.scatterplot(x="setpoint", y="mean_power", data=df, label=label[0], ax=ax[0][1], color=color, marker=marker)
            sns.scatterplot(x="setpoint", y="lower_power", data=df, label=label[0], ax=ax[1][1], color=color, marker=marker)
    for label, df in previous_douglas_fir.groupby(['ignition']):
        color = color_dict['Douglas-fir']
        marker = 'x'
        if not label:
            sns.scatterplot(x="setpoint", y="mean_power", data=df, label='Previous DF', ax=ax[0][0], color=color, marker=marker)
            sns.scatterplot(x="setpoint", y="lower_power", data=df, label='Previous DF', ax=ax[1][0], color=color, marker=marker)
        else:
            sns.scatterplot(x="setpoint", y="mean_power", data=df, label='Previous DF', ax=ax[0][1], color=color, marker=marker)
            sns.scatterplot(x="setpoint", y="lower_power", data=df, label='Previous DF', ax=ax[1][1], color=color, marker=marker)
    ax[1][0].set_xlabel('Heater Setpoint (C)')
    ax[1][1].set_xlabel('Heater Setpoint (C)')
    ax[0][0].set_ylabel('Mean Power (W)')
    ax[1][0].set_ylabel('Lower Power (W)')
    for ax, col in zip(ax[0], ['No Ignition Observed', 'Ignition Observed']):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    lines = []
    labels = []

    for a in fig.axes:
        axLine, axLabel = a.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    # plt.yscale('log')
    fig.legend(set(lines), set(labels), bbox_to_anchor=(1.02, 1), loc=2)

    cp_fluid = 1007
    cp_solid = 2720
    pd.Series({'Douglas-fir': 2720,
               'Oak': 2380,
               'Pine': 2300,
               'Douglas-fir Bark': 1364,
               'Wheat Straw': 1630})
    k_fluid = 26.3E-3
    k_solid = pd.Series({'Douglas-fir': 0.12,
                         'Oak': 0.197,
                         'Pine': 0.110,
                         'Douglas-fir Bark': 0.588,
                         'Wheat Straw': 0.155})
    rho_solid = pd.Series({'Douglas-fir': 510,
                           'Oak': 770,
                           'Pine': 450,
                           'Douglas-fir Bark': 440,
                           'Wheat Straw': 1100})
    rho_fluid = 1.224
    v_container = 9.5E-4

    property_data = test_results.groupby(['species']).density.mean()
    epsilon = 1 - np.divide(property_data, rho_solid)
    k_eff_min = np.divide(1, (1 - epsilon) / k_solid + epsilon / k_fluid)
    k_eff_max = epsilon * k_fluid + (1 - epsilon) * k_solid
    # print(k_eff_min, k_eff_max, abs(k_eff_min-k_eff_max)/k_eff_min*100)
    k_bed = abs(k_eff_min + k_eff_max)/2
    k_bed.name = 'k_bed'
    # cp_bed = mean_density_values[f]/rho_solid*cp_solid
    cp_bed = epsilon * cp_fluid + (1 - epsilon) * cp_solid
    cp_bed.name = 'cp_bed'
    alpha = np.divide(k_bed, property_data * cp_bed)
    alpha.name = 'alpha'
    alpha_min = np.divide(k_eff_min, property_data * cp_bed)
    alpha_max = np.divide(k_eff_max, property_data * cp_bed)
    # print(alpha, k_bed, property_data, cp_bed, epsilon)
    fuel_bed_properties = pd.concat([property_data, cp_bed, k_bed, alpha], axis=1)
