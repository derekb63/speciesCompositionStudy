from __future__ import division
from __future__ import print_function

import pandas as pd
import plotly.express as px
import time
import cantera as ct
import numpy as np
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from logit_response_plots import load_concentration_data
from result_plotting import scrape_data, find_inputs
ct.suppress_thermo_warnings()
print('Runnning Cantera version: ' + ct.__version__)
import matplotlib.pyplot as plt


plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.autolayout'] = True

plt.style.use('ggplot')
plt.style.use('seaborn-pastel')

def ignitionDelay(df, species):
    """
    This function computes the ignition delay from the occurence of the
    peak in species' concentration.
    """
    return df[species].idxmax()


def find_matches_composition(frame, column, text):
    rows = frame[frame[column[0]].str.match(text[0], na=False)]
    rows = rows[rows[column[1]].str.match(text[1], na=False)]
    return rows[['SAMPLE', 'ORGAN FRACTION', 'CELL', 'HCELL', 'LIGH', 'LIGO', 'LIGC', 'TRIL', 'CTANN']]


def reactor_sim(gas, estimated_ignition_delay_time=10):
    r = ct.IdealGasReactor(contents=gas, name='Batch Reactor', energy='off')
    reactorNetwork = ct.ReactorNet([r])
    reactorNetwork.atol = 1e-15
    reactorNetwork.rtol = 1e-9
    # now compile a list of all variables for which we will store data
    stateVariableNames = [r.component_name(item) for item in range(r.n_vars)]
    # use the above list to create a DataFrame
    timeHistory = pd.DataFrame(columns=stateVariableNames)
    t = 0
    # This is a starting estimate. If you do not get an ignition within this time, increase it
    counter = 1
    reactorNetwork.initialize()
    reactorNetwork.max_steps = 1e5
    reactorNetwork.max_err_test_fails = 1e5
    while (t < estimated_ignition_delay_time):
        t = reactorNetwork.step()
        if (counter % 10 == 0):
            # We will save only every 10th value. Otherwise, this takes too long
            # Note that the species concentrations are mass fractions
            #         timeHistory.loc[t] = reactorNetwork.get_state()
            timeHistory.loc[t] = reactorNetwork.get_state()
        counter += 1
    return timeHistory


def stirred_reactor(reactor_gas, residenceTime=100):
    reactorPressure = reactor_gas.P
    pressureValveCoefficient = 0.01
    maxPressureRiseAllowed = 0.01
    maxSimulationTime = 2 * residenceTime  # seconds
    fuelAirMixtureTank = ct.Reservoir(reactor_gas)
    exhaust = ct.Reservoir(reactor_gas)

    stirredReactor = ct.IdealGasReactor(reactor_gas, energy='on')

    massFlowController = ct.MassFlowController(upstream=fuelAirMixtureTank,
                                               downstream=stirredReactor,
                                               mdot=stirredReactor.mass / residenceTime)

    pressureRegulator = ct.Valve(upstream=stirredReactor,
                                 downstream=exhaust,
                                 K=pressureValveCoefficient)

    reactorNetwork = ct.ReactorNet([stirredReactor])
    columnNames = [stirredReactor.component_name(item) for item in range(stirredReactor.n_vars)]
    columnNames = ['pressure', 'density'] + columnNames

    # Use the above list to create a DataFrame
    timeHistory = pd.DataFrame(columns=columnNames)
    # Start the stopwatch
    tic = time.time()

    # Set simulation start time to zero
    t = 0
    counter = 1
    while t < maxSimulationTime:
        t = reactorNetwork.step()

        if (counter % 10 == 0):
            # Extract the state of the reactor
            state = np.hstack([stirredReactor.thermo.P, stirredReactor.thermo.density, stirredReactor.mass,
                               stirredReactor.volume, stirredReactor.T, stirredReactor.thermo.Y])

            # Update the dataframe
            timeHistory.loc[t] = state

        counter += 1

    # Stop the stopwatch
    toc = time.time()

    # We now check to see if the pressure rise during the simulation, a.k.a the pressure valve
    # was okay
    pressureDifferential = timeHistory['pressure'].max() - timeHistory['pressure'].min()

    if (abs(pressureDifferential / reactorPressure) > maxPressureRiseAllowed):
        print("\t WARNING: Non-trivial pressure rise in the reactor. Adjust K value in valve")
    return timeHistory


def stirred_reactor_all_times(reactor_gas, residenceTime=10):
    try:
        reactorPressure = reactor_gas.P
    except AttributeError:
        return np.nan
    pressureValveCoefficient = 0.01
    maxPressureRiseAllowed = 0.01
    maxSimulationTime = 2 * residenceTime  # seconds
    fuelAirMixtureTank = ct.Reservoir(reactor_gas)
    exhaust = ct.Reservoir(reactor_gas)

    stirredReactor = ct.IdealGasReactor(reactor_gas, energy='on')

    massFlowController = ct.MassFlowController(upstream=fuelAirMixtureTank,
                                               downstream=stirredReactor,
                                               mdot=stirredReactor.mass / residenceTime)

    pressureRegulator = ct.Valve(upstream=stirredReactor,
                                 downstream=exhaust,
                                 K=pressureValveCoefficient)

    reactorNetwork = ct.ReactorNet([stirredReactor])
    columnNames = [stirredReactor.component_name(item) for item in range(stirredReactor.n_vars)]
    columnNames = ['pressure', 'density'] + columnNames

    # Use the above list to create a DataFrame
    timeHistory = pd.DataFrame(columns=columnNames)
    # Start the stopwatch
    tic = time.time()

    # Set simulation start time to zero
    t = 0
    counter = 1
    while t < maxSimulationTime:
        t = reactorNetwork.step()

        if (counter % 10 == 0):
            # Extract the state of the reactor
            state = np.hstack([stirredReactor.thermo.P, stirredReactor.thermo.density, stirredReactor.mass,
                               stirredReactor.volume, stirredReactor.T, stirredReactor.thermo.Y])

            # Update the dataframe
            timeHistory.loc[t] = state

        counter += 1

    # Stop the stopwatch
    toc = time.time()

    # We now check to see if the pressure rise during the simulation, a.k.a the pressure valve
    # was okay
    pressureDifferential = timeHistory['pressure'].max() - timeHistory['pressure'].min()

    if (abs(pressureDifferential / reactorPressure) > maxPressureRiseAllowed):
        print("\t WARNING: Non-trivial pressure rise in the reactor. Adjust K value in valve")
    return timeHistory['OH'].idxmax()


def get_equivalence_ratio(fuel, rho_solid, rho_bed):
    epsilon = 1 - rho_bed/rho_solid
    rho_air = 1.204  # kg/m^3
    air_mass_fraction = epsilon * (rho_air / rho_bed)
    air_fuel_bed = air_mass_fraction / (1 - air_mass_fraction)
    air_fuel_stoic = fuel.stoich_air_fuel_ratio(fuel=fuel.mass_fraction_dict(),
                                                oxidizer={'O2': 1.0, 'N2': 3.76},
                                                basis='mass')
    phi = air_fuel_stoic / air_fuel_bed
    return phi


def find_matches_composition(frame, column, text):
    rows = frame[frame[column[0]].str.match(text[0], na=False)]
    rows = rows[rows[column[1]].str.match(text[1], na=False)]
    rows['LIG'] = rows.filter(regex='LIG').sum(axis=1)
    return rows[['SAMPLE', 'ORGAN FRACTION', 'CELL', 'HCELL', 'LIGC', 'LIGH', 'LIGO', 'CTANN', 'TRIL']]


if __name__ == '__main__':
    bio_POx_mech = "/home/derek/Documents/IgnitionProject/BioPOx.cti"
    bio_1412_mech = '/home/derek/Documents/IgnitionProject/bio1412/bio1412.cti'
    solid_species = ct.Solution(bio_POx_mech).species_names
    gaseous_species = ct.Solution(bio_1412_mech).species_names
    common_species = set(solid_species).intersection(gaseous_species)

    # Values for the run
    concentration_data = load_concentration_data()
    # concentration_data = concentration_data.groupby(['SAMPLE', 'ORGAN FRACTION']).mean()
    data_file = '/home/derek/Documents/IgnitionData/SpeciesData/species_data.csv'
    data = pd.read_csv(data_file)
    data['ignition'] = data['ignition'].astype(bool)
    data.groupby('wind_speed')
    material_names = ['Douglas-fir', 'Pine', 'Douglas-fir Bark', 'Oak', 'Wheat Straw', 'Pine Bark']
    composition_names = [['Douglas', 'Untreated Wood'], ['Pine', 'Untreated Wood'], ['Douglas', 'Bark'],
                         ['Oak', 'Untreated Wood'], ['Wheat', 'Straw'], ['Pine', 'Bark']]
    composition_values = pd.DataFrame(index=material_names,
                                      columns=['SAMPLE', 'ORGAN FRACTION', 'CELL', 'HCELL', 'LIGC', 'LIGH', 'LIGO', 'CTANN', 'TRIL'])
    test_result_data = pd.DataFrame(index=material_names, columns=['temperature', 'ignition'])
    for idx, val in enumerate(material_names):
        comp_temp = find_matches_composition(concentration_data, ['SAMPLE', 'ORGAN FRACTION'], composition_names[idx])
        comp_temp['CELL'] = comp_temp['CELL'].astype(float)
        comp_temp['ORGAN FRACTION'] = comp_temp['ORGAN FRACTION'].value_counts().idxmax()
        comp_temp['SAMPLE'] = comp_temp['SAMPLE'].value_counts().idxmax()
        comp_temp[['CELL', 'HCELL', 'LIGC', 'LIGH', 'LIGO', 'CTANN', 'TRIL']] = comp_temp.mean()
        composition_values.loc[val] = comp_temp.iloc[0]

    bowl_volume = 0.00095 # cubic meters
    data_file = '/home/derek/Documents/IgnitionData/SpeciesData/species_data.csv'
    data = pd.read_csv(data_file)
    mass_values = np.multiply(data.groupby(['material']).mean()['mass'], 1e-3)
    density_values = mass_values/bowl_volume
    density_values.name = 'density'
    # density_values = density_values.rename(index={'Douglas-fir': 'Douglas Fir Untreated Wood',
    #                                               'Douglas-fir Bark': 'Douglas Fir Bark',
    #                                               'Oak': 'Oak Untreated Wood',
    #                                               'Pine': 'Pine Untreated Wood',
    #                                               'Wheat Straw': 'Wheat Straw'})
    material_data = composition_values.merge(density_values, left_index=True, right_index=True)
    solid_density = pd.Series({'Douglas-fir': 510,
                               'Oak': 770,
                               'Pine': 450,
                               'Douglas-fir Bark': 440,
                               'Wheat Straw': 1100}, name='solid_density')
    material_data = material_data.merge(solid_density, left_index=True, right_index=True)
    p50_temperatures = pd.Series({'Douglas-fir Bark': 750 + 273,
                                  'Douglas-fir': 574 + 273,
                                  'Oak': 662 + 273,
                                  'Pine': 634 + 273,
                                  'Wheat Straw': 750 + 273}, name='p50_temperature')
    material_data = material_data.merge(p50_temperatures, left_index=True, right_index=True)
    alpha = pd.Series({'Douglas-fir Bark': 5.753943740011115e-07,
                       'Douglas-fir': 2.851289391405382e-07,
                       'Oak': 1.1923060561250857e-07,
                       'Pine': 2.431337661615995e-07,
                       'Wheat Straw': 3.313768321792447e-07}, name='alpha')
    material_data = material_data.merge(alpha, left_index=True, right_index=True)
    phi = dict()
    s_obj = dict()
    key_linker = {'CELL':  'CELL',
                  'HCE':   'HCELL',
                  'LIG-C': 'LIGC',
                  'LIG-H': 'LIGH',
                  'LIG-O': 'LIGO',
                  'CTANN': 'CTANN',
                  'TGL':   'TRIL'}
    test_temperatures = np.flip(np.linspace(523, 1023, 5))
    no_wind_data = material_data.copy(deep=True)
    no_wind_data['p50_temperatures'] = pd.Series({'Douglas-fir': 574+273,
                                        'Pine': 634+273,
                                        'Douglas-fir Bark': 663+273,
                                        'Oak': 750+273,
                                        'Wheat Straw': 750+273})
    for label, row in no_wind_data.iterrows():
        solid_object = ct.Solution(bio_POx_mech)
        solid_object.Y = {label: no_wind_data.iloc[0][val] for label, val in key_linker.items()}
        phi[row.name] = get_equivalence_ratio(solid_object, rho_bed=row['density'], rho_solid=row['solid_density'])
        solid_object.set_equivalence_ratio(phi[row.name],
                                           solid_object.mass_fraction_dict(),
                                           {'O2': 1.0, 'N2': 3.76},
                                           basis='mass')
        solid_object.TP = row['p50_temperature'], ct.one_atm
        s_obj[row.name] = solid_object
    temp_data = no_wind_data.merge(pd.Series(phi, name='phi'), left_index=True, right_index=True)
    temp_data = temp_data.merge(pd.Series(s_obj, name='solid_object'), left_index=True, right_index=True)
    temp_data['sim_results'] = temp_data['solid_object'].apply(reactor_sim)
    temp_data.to_csv('reactor_results_no_wind.csv')



    with_wind_data = material_data.copy(deep=True)
    with_wind_data['p50_temperatures'] = pd.Series({'Douglas-fir': 391+273,
                                                    'Pine': 430+273,
                                                    'Douglas-fir Bark': 750+273,
                                                    'Oak': 750+273,
                                                    'Wheat Straw': 750+273})

    # for reactor_temperature in test_temperatures:
    #     temp_data = material_data.copy(deep=True)
    for label, row in with_wind_data.iterrows():
        solid_object = ct.Solution(bio_POx_mech)
        solid_object.Y = {label: with_wind_data.iloc[0][val] for label, val in key_linker.items()}
        phi[row.name] = get_equivalence_ratio(solid_object, rho_bed=row['density'], rho_solid=row['solid_density'])
        solid_object.set_equivalence_ratio(phi[row.name],
                                           solid_object.mass_fraction_dict(),
                                           {'O2': 1.0, 'N2': 3.76},
                                           basis='mass')
        solid_object.TP = row['p50_temperature'], ct.one_atm
        s_obj[row.name] = solid_object
    temp_data_wind = with_wind_data.merge(pd.Series(phi, name='phi'), left_index=True, right_index=True)
    temp_data_wind = temp_data_wind.merge(pd.Series(s_obj, name='solid_object'), left_index=True, right_index=True)
    temp_data_wind['sim_results'] = temp_data_wind['solid_object'].apply(reactor_sim)
    temp_data_wind.to_csv('reactor_results_with_wind.csv')
    #     species_results[reactor_temperature] = temp_data['sim_results']



