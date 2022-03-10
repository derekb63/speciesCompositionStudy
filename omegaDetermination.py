from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import swifter
matplotlib.use('Qt5Agg')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools as it

# Experimental variables
# Location of the thermocouples relative to the respective smaple rod surface
# tc_locations = np.divide((74.65, 43.33, 26.76, 12.125, -11.01, -26.00, -42.90, -74.62), 1000)
tc_locations = np.divide((74.65, 43.33, 26.76, 12.125, 11.01, 26.00, 42.90, 74.62), 1000)
# Thickness of smaple holder (spacer)
# sample_thickness = 15.30/1000
k_copper = 394
# Diameter of sample holder (m)
sample_diameter = 50e-3
sample_area = np.pi/4 * sample_diameter**2
distances = np.diff(tc_locations)  # Determine the distances between the thermocouples



def retrieve_data_multiindex(folder_path):
    tim_files = []
    for file in os.listdir(folder_path):
        # Use only .tdms files
        if file.endswith('.tdms'):
            tim_files.append(file)
    #sort in order of saving (file numbers increase with time)
    tim_sorted = sorted([(int(os.path.splitext(i)[0].split('_')[-1][-4:-1]), i) for i in tim_files])
    value_frame = pd.DataFrame(columns=['room'] + ["TC{0}".format(x) for x in range(8)])
    # Put all of the data into the previously created array
    for idx, data_file in enumerate(tim_sorted):
        with open(os.path.join(folder_path, data_file[1]), 'rb') as datafile:
            try:
                temp_data = TdmsFile(datafile).as_dataframe().transpose()
                temp_data.columns = ['room'] + ["TC{0}".format(x) for x in range(8)]
                temp_data.reset_index(drop=True, inplace=True)
            except ValueError:
                print(data_file)
        value_frame = value_frame.append(temp_data, ignore_index=True)
    return value_frame


def determine_impedance(sample_temperatures):
    hot_side_temps = sample_temperatures.loc[['TC0', 'TC1', 'TC2']]
    cold_side_temps = sample_temperatures.loc[['TC4', 'TC5', 'TC6', 'TC7']]
    try:
        cold_side = np.polyfit(sorted(tc_locations[:4]), sorted(cold_side_temps.values), 1, full=False)
        hot_side = np.polyfit(sorted(tc_locations[5:]), sorted(hot_side_temps.values), 1, full=False)
        temperature_ratios = np.divide(hot_side[1]-cold_side[1], hot_side[0]+cold_side[0])
        omega = np.multiply((2/k_copper), temperature_ratios)
        return omega
    except ValueError:
        print(sample_temperatures.name)
        return None


def calculate_values(data):
    # data = retrieve_data(folder)
    try:
        no_room_data = data.drop(['room'], axis=1).dropna()
        omega_values = no_room_data.apply(determine_impedance, axis=1)
    except AttributeError:
        omega_values = data.apply(determine_impedance, axis=1)
    return omega_values

def calculate_thermal_conductivity(impedence_values):
    k_sample, r_contact = np.polyfit(x=[15.30E-3, 12.15E-3, 8.2E-3],
                                     y=[impedence_values[15.30],impedence_values[12.15], impedence_values[8.2]],
                                     deg=1)
    return 1/k_sample


def combine_paths(folder):
    species_folder, data_folder = os.path.split(folder)
    if len(folder) < 1:
        return None
    else:
        return os.path.join(root_folder, os.path.join(species_folder, 'TIM_data_'+data_folder))


def plot_impedance_values(impedance_dataframe):
    fig, axs = plt.subplots(3, 1, sharex=True)
    i = 0
    for col in impedance_dataframe.columns.values:
        for line in impedance_dataframe[col].index.values:
            try:
                axs[i].plot(impedance_dataframe[col][line], label=line+"mm")
            except ValueError:
                pass
        axs[i].set_ylabel('Thermal Impedance (K m$^2$/W)')
        axs[i].set_title(col)
        axs[i].set_xlim([0, 15000])
        axs[i].legend()
        i += 1
    axs[i-1].set_xlabel('Time (s)')

    plt.show()

    return None


if __name__ == "__main__":
    window_width = 120
    root_folder = '/home/derek/Documents/IgnitionData/SpeciesData/TIM_data'
    impedance_pickle = os.path.join(root_folder, 'impedance_values2.pkl')

    data_files = [['oak_tests/04_26_2021', 'wheat_tests/04_28_2021'],
                  ['oak_tests/04_23_2021', 'wheat_tests/04_30_2021'],
                  ['oak_tests/04_22_2021', 'wheat_tests/05_04_2021']]
    # data_files = [['03_12_2020', '03_13_2020'],
    #               ['03_14_2020', '03_16_2020'],
    #               ['03_15_2020', '03_17_2020']]

    # folder_names = pd.DataFrame(data_files,
    #                             index=['15.30', '12.15', '8.2'],
    #                             columns=['#18 granulated', '0.125in screened', '0.25in screened'])
    folder_names = pd.DataFrame(data_files,
                                index=['15.30', '12.15', '8.2'],
                                columns=['oak_wood', 'wheat_straw'])
    # Convert the folder names to a muliindex data frame
    folder_index = pd.MultiIndex.from_product([folder_names.index, folder_names.columns])
    folder_series = pd.Series(index=folder_index, dtype=str)
    for i in folder_series.index:
        folder_series[i] = folder_names.loc[i]
    # Rename folder_names to be a series with the multiindex and complete filenames
    folder_names = folder_series.apply(combine_paths)

    raw_values = pd.read_csv('raw_values.csv', index_col=[0, 1, 2])

    # frame_list = []
    # for idx, folder_str in folder_names.items():
    #     temp_frame = retrieve_data_multiindex(folder_str)
    #     index_values = pd.MultiIndex.from_tuples([idx + tuple([x]) for x in temp_frame.index])
    #     temp_frame.set_index(index_values, inplace=True)
    #     temp_frame.reset_index()
    #     frame_list.append(temp_frame)
    # raw_values = pd.concat(frame_list)
    impedance_values = raw_values.reset_index().swifter.apply(determine_impedance, axis=1)
    impedance_values.index = raw_values.index
    raw_values = raw_values.assign(impedence=impedance_values)
    data_collection = pd.DataFrame(columns=raw_values.index.get_level_values('species').unique())
    for material in raw_values.index.get_level_values('species').unique():
        temp_frame = pd.DataFrame.from_dict({15.3: raw_values.loc[15.3, material, slice(None)].impedence,
                                             12.15: raw_values.loc[12.15, material, slice(None)].impedence,
                                             8.2: raw_values.loc[8.2, material, slice(None)].impedence},
                                            )
        data_collection[material] = temp_frame.dropna().apply(calculate_thermal_conductivity, axis=1)

    plt.figure()
    data_collection.iloc[:50000].plot()
    plt.xlabel('Time (s)')
    plt.ylabel('Thermal Conductivity (W m<sup>-1</sup> K<sup>-1</sup>)')
    plt.ylim([0, 5])
    plt.show()




    '''
    # try:
    #     data_collection = pd.read_pickle(impedance_pickle)
    # except FileNotFoundError:
    #     # data_collection = pd.DataFrame(index=['15.30', '12.15', '8.2'],
    #     #                                columns=['#18 granulated', '0.125in screened', '0.25in screened'])
    #     data_collection = pd.DataFrame(index=['15.30', '12.15', '8.2'],
    #                                    columns=['oak_wood', 'wheat_straw'])
    data_collection = raw_values.applymap(calculate_values)
    #     data_collection = folder_names.applymap(calculate_values)
    #     # data_collection.to_pickle(impedance_pickle)
    #
    # # plot_impedance_values(data_collection)



    combined_fig = make_subplots(rows=6, cols=3, start_cell="top-left",
                                 specs=[[{}, {"rowspan": 3}, {"rowspan": 3}],
                                        [{}, None, None],
                                        [{}, None, None],
                                        [{}, {"rowspan": 3}, {"rowspan": 3}],
                                        [{}, None, None],
                                        [{}, None, None]
                                        ],
                                 column_titles=['Temperature Values', 'Thermal Impedance', 'Thermal Conductivity'],
                                 row_titles=['Oak Wood', 'Wheat Straw'],
                                 shared_xaxes=True)
    row = 1
    colors = it.cycle(['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF'])
    for size in data_collection.columns:
        for thickness in data_collection.index:
            if row == 1:
                legend_switch = True
            else:
                legend_switch = False
            temp_data = raw_values[size][thickness]
            for label, content in temp_data.iteritems():
                combined_fig.add_trace(go.Scattergl(x=content.index,
                                                    y=content.values,
                                                    mode='lines',
                                                    line_color=next(colors),
                                                    name=str(label),
                                                    legendgroup=label,
                                                    showlegend=legend_switch),
                                       row=row,
                                       col=1)
            row += 1
    row = 1
    colors = it.cycle(['#AA0DFE', '#1C8356', '#F6222E'])
    for size in data_collection.columns:
        for thickness in data_collection.index:
            if row == 1:
                legend_switch = True
            else:
                legend_switch = False
            combined_fig.add_trace(go.Scattergl(x=data_collection[size][thickness].index,
                                                y=data_collection[size][thickness].values,
                                                mode='lines',
                                                line_color=next(colors),
                                                name=str(thickness) + 'mm impedance',
                                                legendgroup=thickness,
                                                showlegend=legend_switch),
                                   row=row,
                                   col=2)
        row += 3
    row = 1
    for size in data_collection.columns:
        if row == 1:
            legend_switch = True
        else:
            legend_switch = False
        k_value_data = data_collection[size].values
        min_length = test_len = min([len(x) for x in k_value_data])
        reshape_array = np.empty((len(k_value_data), min_length))
        for idx, val in enumerate(k_value_data):
            reshape_array[idx, :min_length] = val[:min_length]
        reshape_array = reshape_array.transpose()
        k_results = np.apply_along_axis(func1d=calculate_thermal_conductivity, arr=reshape_array, axis=1)
        combined_fig.add_trace(go.Scattergl(x=data_collection[size]['8.2'].index,
                                            y=[-k[0] for k in k_results],
                                            mode='lines',
                                            line_color='#222A2A',
                                            legendgroup='sigma',
                                            showlegend=legend_switch,
                                            name='&#963;'),
                               row=row,
                               col=3)
        row += 3
    combined_fig.update_xaxes(range=[0, 10000])
    combined_fig.update_xaxes(title_text='Time (s)', row=9)
    combined_fig.update_yaxes(title_text="Temperature (<sup>o</sup>C)", row=5, col=1)
    combined_fig.update_yaxes(title_text='Thermal Impedance (K m<sup>2</sup>W<sup>-1</sup>)', col=2)
    combined_fig.update_yaxes(title_text='Thermal Conductivity (W m<sup>-1</sup> K<sup>-1</sup>)', col=3)
    # combined_fig.update_yaxes(range=[0, 10], col=3)
    # combined_fig.update_yaxes(range=[0, 60], col=1)
    combined_fig.write_html('/home/derek/Desktop/impedance_species.html')
    '''
