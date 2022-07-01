import os
import pandas as pd
import numpy as np
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile


def column_polyfit(column_values):
    fit = np.polyfit(column_values.index, column_values.values, deg=2)
    return fit

if __name__ == "__main__":
    filename = '/home/derek/PycharmProjects/speciesCompositionStudy/1DSim_Data/pine_bark_low_wind.xlsx'
    test_data = pd.read_excel(filename, index_col=0)
    # # Convert mass fraction to mass flux for each species
    # flux_data = test_data.multiply(test_data['MASS FLUX'], axis=0).drop('MASS FLUX', axis=1)

    # fit all of the data to second order polynomials
    fit_results = test_data.apply(column_polyfit, axis=0, raw=False)

    directory_for_setup = '/media/derek/StorageDisk/speciesStudyFoam/paperCases/quiescent/speciesConcentrations'
    example_file = 'C7H16_example'

    species_mapping_dict = {'PHENOL': 'C6H5OH',
                            'HMFU': 'C6H6O3',
                            'COUMARYL': 'C9H10O2',
                            'ANISOLE': 'C6H5OCH3',
                            'ACROLEIN': 'C2H3CHO'
                            }
    # convert the mass flux from Diba's caclualtions to mass flow rate for simulations
    injectionArea = 2e-6 # measured exit area for the 2D simulation cases
    test_data['MASS FLUX'] = test_data['MASS FLUX']*injectionArea

    for column_name, column_data in test_data.iloc[:].iteritems():
        f = ParsedParameterFile(os.path.join(directory_for_setup, example_file))
        if column_name == 'TEMPERATURE':
            column_name = 'T'
        elif column_name == 'MASS FLUX':
            column_name = 'U'
            f['boundaryField']['injectionZones']['type'] = 'flowRateInletVelocity'
        elif column_name in species_mapping_dict:
            column_name = species_mapping_dict[column_name]
        else:
            column_name = column_name
        print(column_name)
        f.header['object'] = str(column_name)
        time_values = column_data.index.to_numpy() - 0.01
        mass_fraction_values = column_data.to_numpy()
        time_mass_fraction_table = list(zip(time_values, mass_fraction_values))
        time_mass_fraction_table = [str(x).replace(',', '') for x in time_mass_fraction_table]
        f['boundaryField']['injectionZones']['uniformValue'][1] = time_mass_fraction_table
        destination_file = os.path.join(directory_for_setup, str(column_name))
        f.writeFileAs(destination_file)

   # f['boundaryField']['gasInlets']['value'] = 'uniform {0}'.format(bio_gas.mass_fraction_dict()[species])
   # f.header['object'] = str(species)
   # if species == 'N2':
   #     f['internalField'] = 'uniform 0.76699'
   # elif species == 'O2':
   #     f['internalField'] = 'uniform .23301'
   # open(destination_file, 'a+').close()
   # f.writeFileAs(destination_file)
   # print(species)