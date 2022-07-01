from __future__ import division
from __future__ import print_function

import pandas as pd
import plotly.express as px
import time
import cantera as ct
import numpy as np
import seaborn
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


def intersection_gas_object(row_values):
    intersection_species = set(ct.Solution(bio_POx_mech).species_names).intersection(
        ct.Solution(bio_1412_mech).species_names)
    gas_object = ct.Solution(bio_1412_mech)
    gas_object.Y = row_values.sim_results[intersection_species].iloc[-1].to_dict()
    gas_object.TP = row_values.sim_results['temperature'].iloc[-1], ct.one_atm
    gas_object.set_equivalence_ratio(1, gas_object.mole_fraction_dict(), {'O2': 1, 'N2': 3.76})
    return gas_object


def intersection_gas_object_all_times(row_values):
    intersection_species = set(ct.Solution(bio_POx_mech).species_names).intersection(
        ct.Solution(bio_1412_mech).species_names)
    gas_series = pd.Series(dtype='object')
    for idx_time, val in row_values.sim_results.iloc[::10].iterrows():
        if idx_time < 2:
            gas_object = ct.Solution(bio_1412_mech)
            gas_object.Y = val[intersection_species].to_dict()
            gas_object.TP = val['temperature'], ct.one_atm
            try:
                gas_object.set_equivalence_ratio(1, gas_object.mole_fraction_dict(), {'O2': 1, 'N2': 3.76})
                gas_series[idx_time] = gas_object
            except ct.CanteraError:
                pass
    return gas_series


def ignition_delay(simulation_results, species='OH'):
    species_time = simulation_results[species].idxmax()
    return species_time


def find_matches_composition(frame, column, text):
    rows = frame[frame[column[0]].str.match(text[0], na=False)]
    rows = rows[rows[column[1]].str.match(text[1], na=False)]
    rows['LIG'] = rows.filter(regex='LIG').sum(axis=1)
    return rows[['SAMPLE', 'ORGAN FRACTION', 'CELL', 'HCELL', 'LIGC', 'LIGH', 'LIGO', 'CTANN', 'TRIL']]

ambient_loss_curve_fit = np.poly1d([1.70673855e-04, -3.12525005e-01,  1.46914196e+02])
heater_diameter = 0.00635  # meters
heater_length = 0.040  # meters
# for this analysis the ends of the heater will not be considered since the
heater_area = np.pi * heater_diameter * heater_length

if __name__ == '__main__':
    bio_POx_mech = "/home/derek/Documents/IgnitionProject/BioPOx.cti"
    bio_1412_mech = '/home/derek/Documents/IgnitionProject/bio1412/bio1412.cti'
    solid_species = ct.Solution(bio_POx_mech).species_names
    gaseous_species = ct.Solution(bio_1412_mech).species_names
    common_species = set(solid_species).intersection(gaseous_species)

    # Look for the output files and load the data in the files to a DataFrame
    # data_directory = '/home/derek/Documents/IgnitionData/SpeciesData/'
    # data_files = find_inputs(data_directory)
    # test_results = pd.DataFrame(map(scrape_data, data_files))


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
    # add a column to state if ignition happened
    # test_results['ignition'] = ~test_results['time_to_ignition'].isna()
    # # Group the data based on the species and ignition
    # density_values = test_results.groupby('species').density.mean()
    composition_values = composition_values.rename(index={'Douglas-fir': 'Douglas Fir Untreated Wood',
                                                  'Douglas-fir Bark': 'Douglas Fir Bark',
                                                  'Oak': 'Oak Untreated Wood',
                                                  'Pine': 'Pine Untreated Wood',
                                                  'Wheat Straw': 'Wheat Straw'})
    composition_values.to_csv('/home/derek/PycharmProjects/speciesCompositionStudy/material_composition.csv')
    density_values = density_values.rename(index={'Douglas-fir': 'Douglas Fir Untreated Wood',
                                                  'Douglas-fir Bark': 'Douglas Fir Bark',
                                                  'Oak': 'Oak Untreated Wood',
                                                  'Pine': 'Pine Untreated Wood',
                                                  'Wheat Straw': 'Wheat Straw'})
    material_data = composition_values.merge(density_values, left_index=True, right_index=True)
    solid_density = pd.Series({'Douglas Fir Untreated Wood': 510,
                               'Oak Untreated Wood': 770,
                               'Pine Untreated Wood': 450,
                               'Douglas Fir Bark': 440,
                               'Wheat Straw': 1100}, name='solid_density')
    material_data = material_data.merge(solid_density, left_index=True, right_index=True)
    p50_temperatures = pd.Series({'Douglas Fir Bark': 750 + 273,
                                  'Douglas Fir Untreated Wood': 574 + 273,
                                  'Oak Untreated Wood': 662 + 273,
                                  'Pine Untreated Wood': 634 + 273,
                                  'Wheat Straw': 750 + 273}, name='p50_temperature')
    material_data = material_data.merge(p50_temperatures, left_index=True, right_index=True)
    exit_width = pd.Series({'Douglas Fir Bark': 0.00375,
                                  'Douglas Fir Untreated Wood': .00025,
                                  'Oak Untreated Wood': np.nan,
                                  'Pine Untreated Wood': 0.00385,
                                  'Wheat Straw': 0.0018}, name='exit_width')
    material_data = material_data.merge(exit_width, left_index=True, right_index=True)
    hot_volume = pd.Series({'Douglas Fir Bark': 3.73E-9,
                                  'Douglas Fir Untreated Wood': 1.202E-9,
                                  'Oak Untreated Wood': np.nan,
                                  'Pine Untreated Wood': 3.75E-9,
                                  'Wheat Straw': 2.13E-8}, name='hot_volume')
    material_data = material_data.merge(hot_volume, left_index=True, right_index=True)
    heat_flux = pd.Series({'Douglas Fir Bark': 15560,
                                  'Douglas Fir Untreated Wood': 6110,
                                  'Oak Untreated Wood': 10160,
                                  'Pine Untreated Wood': 8300,
                                  'Wheat Straw': 15560}, name='heat_flux')
    material_data = material_data.merge(heat_flux, left_index=True, right_index=True)
    alpha = pd.Series({'Douglas Fir Bark': 5.753943740011115e-07,
                       'Douglas Fir Untreated Wood': 2.851289391405382e-07,
                       'Oak Untreated Wood': 1.1923060561250857e-07,
                       'Pine Untreated Wood': 2.431337661615995e-07,
                       'Wheat Straw': 3.313768321792447e-07}, name='alpha')
    material_data = material_data.merge(alpha, left_index=True, right_index=True)
    # material_data['bed_flux'] = material_data['heat_flux']-\
    #                         (ambient_loss_curve_fit(material_data['p50_temperature']))/(0.5 * heater_area)
    phi = dict()
    s_obj = dict()
    key_linker = {'CELL':  'CELL',
                  'HCE':   'HCELL',
                  'LIG-C': 'LIGC',
                  'LIG-H': 'LIGH',
                  'LIG-O': 'LIGO',
                  'CTANN': 'CTANN',
                  'TGL':   'TRIL'}

    for label, row in material_data.iterrows():
        solid_object = ct.Solution(bio_POx_mech)
        solid_object.Y = {label: material_data.iloc[0][val] for label, val in key_linker.items()}
        phi[row.name] = get_equivalence_ratio(solid_object, rho_bed=row['density'], rho_solid=row['solid_density'])
        solid_object.set_equivalence_ratio(phi[row.name],
                                           solid_object.mass_fraction_dict(),
                                           {'O2': 1.0, 'N2': 3.76},
                                           basis='mass')
        solid_object.TP = row['p50_temperature'], ct.one_atm
        s_obj[row.name] = solid_object
    material_data = material_data.merge(pd.Series(phi, name='phi'), left_index=True, right_index=True)
    material_data = material_data.merge(pd.Series(s_obj, name='solid_object'), left_index=True, right_index=True)
    material_data['sim_results'] = material_data['solid_object'].apply(reactor_sim)

    # material_data['gas_object'] = material_data.apply(intersection_gas_object_all_times, axis=1)
    gas_object_time_frame = material_data.apply(intersection_gas_object_all_times, axis=1).transpose()
    tau_values = gas_object_time_frame.applymap(stirred_reactor_all_times)
    # material_data['gas_results'] = material_data['gas_object'].apply(reactor_sim)
    # material_data['gas_results'] = material_data['gas_object'].apply(stirred_reactor_all_times)
    # material_data['tau'] = material_data['gas_results'].apply(ignition_delay)
    # material_data['gas_fraction'] = material_data['sim_results'].apply(lambda x: x[common_species].iloc[-1].sum())
    # material_data['exit_velocity'] = np.divide((material_data['hot_volume'] *
    #                                             material_data['solid_density'] *
    #                                             material_data['gas_fraction']),
    #                                            material_data['gas_results'].apply(lambda x: x['density'].iloc[-1]) *
    #                                            material_data['exit_width'].mul(20*0.001))
    # material_data['Da_y'] = np.divide(0.006, material_data['exit_velocity'] * material_data['tau'])
    # material_data['Da_x'] = material_data['bed_flux']*material_data['alpha']
    # fig = px.scatter(material_data, x='Da_x', y='Da_y', hover_data=['SAMPLE', 'ORGAN FRACTION'])
    # fig.write_html('da_plot.html')
    #
    # flame_temperature_plot = make_subplots(rows=len(material_data['sim_results'].index),
    #                                        cols=1,
    #                                        subplot_titles=material_data['sim_results'].index.values)
    # idx = 1
    # for name, gas_states in material_data['sim_results'].items():
    #     # Create new lines in the plot for the species of interest
    #     for species in gas_states.iloc[-1][common_species].index:
    #         flame_temperature_plot.add_trace(go.Scatter(x=gas_states.index,
    #                                                     y=gas_states[species],
    #                                                     name=species,
    #                                                     mode='lines',
    #                                                     legendgroup=species,
    #                                                     showlegend=not(bool(idx-1))),
    #                                          row=idx, col=1
    #                                          )
    #     # Add titles and axis labels as well as a legend title
    #     flame_temperature_plot.update_layout(title=name,
    #                                          xaxis_title='Time (s)',
    #                                          yaxis_title='Mass Fraction (Y)',
    #                                          legend_title='Species'
    #                                          )
    #     flame_temperature_plot.update_xaxes(range=[0, 5])
    #     idx += 1
    # flame_temperature_plot.write_html('solid_concentrations.html')
    #
    # flame_temperature_plot = make_subplots(rows=len(material_data['sim_results'].index),
    #                                        cols=1,
    #                                        subplot_titles=material_data['sim_results'].index.values)
    # idx = 1
    # for name, gas_states in material_data['gas_results'].items():
    #
    #     # Create new lines in the plot for the species of interest
    #     for species in gas_states.iloc[-1][common_species].index:
    #         flame_temperature_plot.add_trace(go.Scatter(x=gas_states.index,
    #                                                     y=gas_states[species],
    #                                                     name=species,
    #                                                     mode='lines',
    #                                                     legendgroup=species,
    #                                                     showlegend=not(bool(idx-1))),
    #                                          row=idx, col=1
    #                                          )
    #     # Add titles and axis labels as well as a legend title
    #     flame_temperature_plot.update_layout(title=name,
    #                                          xaxis_title='Time (s)',
    #                                          yaxis_title='Mass Fraction (Y)',
    #                                          legend_title='Species'
    #                                          )
    #     flame_temperature_plot.update_xaxes(range=[0, 1.25])
    #     idx += 1
    # flame_temperature_plot.write_html('gas_concentrations.html')
    #
    # cp_fluid = 1007
    # cp_solid = 2720
    # pd.Series({'Douglas-fir': 2720,
    #            'Oak': 2380,
    #            'Pine': 2300,
    #            'Douglas-fir Bark': 1364,
    #            'Wheat Straw': 1630})
    # k_fluid = 26.3E-3
    # k_solid = pd.Series({'Douglas-fir': 0.12,
    #                      'Oak': 0.197,
    #                      'Pine': 0.110,
    #                      'Douglas-fir Bark': 0.588,
    #                      'Wheat Straw': 0.155})
    # rho_solid = pd.Series({'Douglas-fir': 510,
    #                        'Oak': 770,
    #                        'Pine': 450,
    #                        'Douglas-fir Bark': 440,
    #                        'Wheat Straw': 1100})
    # rho_fluid = 1
    # fourier_max = 0.075
    # v_container = 9.5E-4
    #
    # property_data = test_results.groupby(['species']).density.mean()
    # epsilon = 1 - np.divide(property_data, rho_solid)
    # k_eff_min = np.divide(1, (1 - epsilon) / k_solid + epsilon / k_fluid)
    # k_eff_max = epsilon * k_fluid + (1 - epsilon) * k_solid
    # # print(k_eff_min, k_eff_max, abs(k_eff_min-k_eff_max)/k_eff_min*100)
    # k_bed = abs(k_eff_min + k_eff_max)/2
    # k_bed.name = 'k_bed'
    # # cp_bed = mean_density_values[f]/rho_solid*cp_solid
    # cp_bed = epsilon * cp_fluid + (1 - epsilon) * cp_solid
    # cp_bed.name = 'cp_bed'
    # alpha = np.divide(k_bed, property_data * cp_bed)
    # alpha.name = 'alpha'
    # alpha_min = np.divide(k_eff_min, property_data * cp_bed)
    # alpha_max = np.divide(k_eff_max, property_data * cp_bed)
    # # print(alpha, k_bed, property_data, cp_bed, epsilon)
    # fuel_bed_properties = pd.concat([property_data, cp_bed, k_bed, alpha], axis=1)
    #
    #
