import time
import pandas as pd
import numpy as np
import cantera as ct

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.style.use("seaborn-pastel")

plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 120


def column_polyfit(column_values):
    fit = np.polyfit(column_values.index, column_values.values, deg=2)
    return fit

if __name__ == "__main__":
    filename = '/home/derek/PycharmProjects/speciesCompositionStudy/1DSim_Data/douglas_fir_wood_low_wind.xlsx'
    test_data = pd.read_excel(filename, index_col=0)
    # # Convert mass fraction to mass flux for each species
    # flux_data = test_data.multiply(test_data['MASS FLUX'], axis=0).drop('MASS FLUX', axis=1)

    # fit all of the data to second order polynomials
    fit_results = test_data.apply(column_polyfit, axis=0, raw=False)

    species_mapping_dict = {'PHENOL': 'C6H5OH',
                            'HMFU': 'C6H6O3',
                            'COUMARYL': 'C9H10O2',
                            'ANISOLE': 'C6H5OCH3',
                            'ACROLEIN': 'C2H3CHO'
                            }
    test_data.rename(columns=species_mapping_dict, inplace=True)
    ################################### Cantera Perfectly Stirred Reactor #############################################
    gas = ct.Solution('/home/derek/Documents/IgnitionProject/bio1412/bio1412.cti')
    reactor_temperature = 1023
    reactor_pressure = ct.one_atm
    inlet_concentrations = test_data.drop(['MASS FLUX', 'TEMPERATURE', 'LINOLEIC ACID'], axis=1).iloc[0]

    gas.TPX = reactor_temperature, reactor_pressure, inlet_concentrations.to_dict()

    residence_time = 3  # seconds
    reactor_volume = 2.16e-7  # cubic meters

    max_simulation_time = 100 # seconds

    fuel_air_mixture_tank = ct.Reservoir(gas)
    exhaust = ct.Reservoir(gas)

    stirred_reactor = ct.IdealGasReactor(gas, energy="off", volume=reactor_volume)

    mass_flow_controller = ct.MassFlowController(
        upstream=fuel_air_mixture_tank,
        downstream=stirred_reactor,
        mdot=stirred_reactor.mass / residence_time,
    )

    pressure_regulator = ct.PressureController(
        upstream=stirred_reactor, downstream=exhaust, master=mass_flow_controller
    )

    reactor_network = ct.ReactorNet([stirred_reactor])

    # Create a SolutionArray to store the data
    time_history = ct.SolutionArray(gas, extra=["t"])

    # Start the stopwatch
    tic = time.time()

    # Set simulation start time to zero
    t = 0
    counter = 1
    while t < max_simulation_time:
        t = reactor_network.step()

        # We will store only every 10th value. Remember, we have 1200+ species, so there will be
        # 1200+ columns for us to work with
        time_history.append(stirred_reactor.thermo.state, t=t)

        counter += 1

    # Stop the stopwatch
    toc = time.time()

    print(f"Simulation Took {toc - tic:3.2f}s to compute, with {counter} steps")

    plt.figure()
    plt.semilogx(time_history.t, time_history("C6H5OH").X, "-o")
    plt.xlabel("Time (s)")
    plt.ylabel("Mole Fraction : $X_{PHENOL}$");


