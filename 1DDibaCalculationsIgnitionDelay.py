import cantera as ct
import pandas as pd
import time
import numpy as np
import swifter

def ignition_delay(states, species):
    """
    This function computes the ignition delay from the occurence of the
    peak in species' concentration.
    """
    i_ign = states(species).Y.argmax()
    return states.t[i_ign]


def reactor_calculation(reactor_data):
    gas = ct.Solution('/home/derek/Documents/IgnitionProject/bio1412/bio1412.cti')
    reference_species = "C6H5OH"

    gas.TP = reactor_data['TEMPERATURE'], reactor_data['PRESSURE']
    gas.Y = reactor_data.drop(['TEMPERATURE', 'LINOLEIC ACID', 'PRESSURE', 'level_0']).to_dict()

    r = ct.IdealGasReactor(contents=gas, name="Batch Reactor")
    reactor_network = ct.ReactorNet([r])

    # use the above list to create a DataFrame
    time_history = ct.SolutionArray(gas, extra="t")

    # Tic
    t0 = time.time()

    # This is a starting estimate. If you do not get an ignition within this time, increase it
    estimated_ignition_delay_time = 1000
    t = 0

    counter = 1
    while t < estimated_ignition_delay_time:
        t = reactor_network.step()
        # if not counter % 10:
        # We will save only every 10th value. Otherwise, this takes too long
        # Note that the species concentrations are mass fractions
        time_history.append(r.thermo.state, t=t)
        counter += 1

    # We will use the 'oh' species to compute the ignition delay
    tau = ignition_delay(time_history, reference_species)

    # Toc
    t1 = time.time()

    # print(f"Computed Ignition Delay: {tau:.3e} seconds. Took {t1 - t0:3.2f}s to compute.")

    return tau
if __name__ == "__main__":
    filenames = {"oak_wood": '/home/derek/PycharmProjects/speciesCompositionStudy/1DSim_Data/oak_wood_low_wind.xlsx',
                 "df_wood": '/home/derek/PycharmProjects/speciesCompositionStudy/1DSim_Data/douglas_fir_wood_low_wind.xlsx',
                 "pine_bark": '/home/derek/PycharmProjects/speciesCompositionStudy/1DSim_Data/pine_bark_low_wind.xlsx',
                 "pine_wood": '/home/derek/PycharmProjects/speciesCompositionStudy/1DSim_Data/pine_wood_low_wind.xlsx'}

    result_data = pd.DataFrame(columns=filenames.keys())
    data_list = dict()
    for key, value in filenames.items():
        test_data = pd.read_excel(value, index_col=0)
        species_mapping_dict = {'PHENOL': 'C6H5OH',
                                'HMFU': 'C6H6O3',
                                'COUMARYL': 'C9H10O2',
                                'ANISOLE': 'C6H5OCH3',
                                'ACROLEIN': 'C2H3CHO'
                                }
        test_data.rename(species_mapping_dict, inplace=True, axis=1)
        test_data['PRESSURE'] = ct.one_atm
        data_list[key] = test_data
    combined_data = pd.concat(data_list.values(), keys=data_list.keys())
    combined_data = combined_data.loc[:, ~combined_data.columns.str.startswith('MASS')]
    combined_data = combined_data.reset_index().drop('TIME (s)', axis=1)
    combined_result = combined_data.swifter.apply(reactor_calculation, axis=1)
    oak_wood = combined_result[:1001]
    oak_wood.name = 'oak_wood'
    df_wood = combined_result[1001:2002]
    df_wood.name = 'df_wood'
    pine_bark = combined_result[2002:3003]
    pine_bark.name = 'pine_bark'
    pine_wood = combined_result[3003:]
    pine_wood.name = 'pine_wood'
    # result_data = pd.concat([x.reset_index().drop('index', axis=1) for x in [oak_wood, df_wood, pine_bark, pine_wood]], axis=1)
    # result_data.set_index(np.arange(0, 10.01, 0.01), inplace=True)
        # result_data[key] = test_data.swifter.apply(reactor_calculation, axis=1)
    # If you want to save all the data - molefractions, temperature, pressure, etc
    # uncomment the next line
    # time_history.to_csv("time_history.csv")

