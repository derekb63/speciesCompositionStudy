import pandas as pd
import cantera as ct
import time
import numpy as np
import swifter
ct.suppress_thermo_warnings()




def load_concentration_data(species_data_file='/home/derek/Documents/IgnitionProject/ef5b01753_si_001.csv'):
    with open(species_data_file, 'r') as f:
        concentration_frame = pd.read_csv(f)
    concentration_frame = concentration_frame[[x for x in concentration_frame.columns if 'unnamed' not in x.lower()]]
    concentration_frame.drop(
        columns=['TAXONOMY', 'REF.', 'beta', 'gamma', 'delta', 'epsion', 'alpha', 'VARIETY', 'N', 'S', 'Cl', 'Ash'],
        inplace=True)
    concentration_frame.rename(columns={concentration_frame.columns[0]: 'CATEGORY'}, inplace=True)
    return concentration_frame


def exit_ignition_delay(fuel_composition, reactorTemperature=750):
    mechFile = "/home/derek/Documents/IgnitionProject/BioPOx.cti"
    volume_solid = 5e-8  # m**3
    l = 0.001
    w = 0.001
    fuel_composition = fuel_composition[['CELL', 'HCELL', 'LIGH', 'LIGO', 'LIGC', 'TRIL', 'CTANN']]
    fuel_dict = {'CELL': fuel_composition['CELL'],
                 'HCE': fuel_composition['HCELL'],
                 'LIG-C': fuel_composition['LIGC'],
                 'LIG-H': fuel_composition['LIGH'],
                 'LIG-O': fuel_composition['LIGO'],
                 'CTANN': fuel_composition['CTANN'],
                 'TGL': fuel_composition['TRIL']}
    try:
        fuel_dict = {k: float(d) for k, d in fuel_dict.items()}
    except ValueError:
        return dict(ignition_delay=None, transit_distance=None)
    gas = ct.Solution(mechFile)
    gas.set_equivalence_ratio(phi=1000,
                              fuel=fuel_dict,
                              oxidizer={'O2': 1.0, 'N2': 3.76})
    gas.TP = reactorTemperature, ct.one_atm
    # Create a batch reactor object and add it to a reactor network
    # In this example, the batch reactor will be the only reactor
    # in the network
    r = ct.IdealGasReactor(contents=gas, name='Batch Reactor', energy='off')
    # r = ct.ConstPressureReactor(contents=gas, name="Batch Reactor", energy='on',volume=reactorVolume)
    # hot_side = ct.Reservoir(gas)
    # f1 = ct.Func1(lambda t: 5)
    # wall = ct.Wall(left=hot_side, right=r, Q=f1)
    # wall.set_heat_flux(f1)

    reactorNetwork = ct.ReactorNet([r])
    reactorNetwork.atol = 1e-20
    reactorNetwork.rtol = 1e-10
    # now compile a list of all variables for which we will store data
    stateVariableNames = [r.component_name(item) for item in range(r.n_vars)]

    # use the above list to create a DataFrame
    timeHistory = pd.DataFrame(columns=stateVariableNames)
    # Tic
    t0 = time.time()

    # This is a starting estimate. If you do not get an ignition within this time, increase it
    estimatedIgnitionDelayTime = 10
    t = 0

    counter = 1;
    while (t < estimatedIgnitionDelayTime):
        t = reactorNetwork.step()
        if (counter % 10 == 0):
            # We will save only every 10th value. Otherwise, this takes too long
            # Note that the species concentrations are mass fractions
            #         timeHistory.loc[t] = reactorNetwork.get_state()
            timeHistory.loc[t] = reactorNetwork.get_state()
        counter += 1
    # Toc
    t1 = time.time()

    bio_1412_file = '/home/derek/Documents/IgnitionProject/bio1412/bio1412.cti'
    bio_gas = ct.Solution(bio_1412_file)
    intersection_species = set(bio_gas.species_names).intersection(gas.species_names)
    bio_gas.Y = timeHistory[intersection_species].iloc[-1].to_dict()
    bio_gas.TP = reactorTemperature, ct.one_atm
    mass_fraction_gas = timeHistory[intersection_species].iloc[-1].values.sum()
    mass_gas = volume_solid * 135 * mass_fraction_gas
    v_exit = np.divide((volume_solid * 135 * mass_fraction_gas),
                       (20 * bio_gas.density_mass * l * w))
    residenceTime = 100  # s

    # Instrument parameters

    # This is the "conductance" of the pressure valve and will determine its efficiency in
    # holding the reactor pressure to the desired conditions.
    pressureValveCoefficient = 0.01

    # This parameter will allow you to decide if the valve's conductance is acceptable. If there
    # is a pressure rise in the reactor beyond this tolerance, you will get a warning
    maxPressureRiseAllowed = 0.01
    # Simulation termination criterion
    maxSimulationTime = 2 * residenceTime  # seconds
    fuelAirMixtureTank = ct.Reservoir(bio_gas)
    exhaust = ct.Reservoir(bio_gas)

    stirredReactor = ct.IdealGasReactor(bio_gas, energy='on')

    massFlowController = ct.MassFlowController(upstream=fuelAirMixtureTank,
                                               downstream=stirredReactor,
                                               mdot=stirredReactor.mass / residenceTime)

    pressureRegulator = ct.Valve(upstream=stirredReactor,
                                 downstream=exhaust,
                                 K=pressureValveCoefficient)
    reactorNetwork = ct.ReactorNet([stirredReactor])
    # reactorNetwork.atol = 1e-10
    # Now compile a list of all variables for which we will store data
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
    ignitionDelayExit = timeHistory['OH'].idxmax()
    if (abs(pressureDifferential / ct.one_atm) > maxPressureRiseAllowed):
        print("\t WARNING: Non-trivial pressure rise in the reactor. Adjust K value in valve")

    return dict(ignition_delay=ignitionDelayExit,
                transit_distance=(ignitionDelayExit * v_exit),
                concentrations=timeHistory)




if __name__ == '__main__':
    species_composition = load_concentration_data()
    # return_values = species_composition.apply(exit_ignition_delay, axis=1)




