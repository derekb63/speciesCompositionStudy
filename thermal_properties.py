import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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

solid_density_values = {'Douglas-fir': 480,  # Values except wheat straw taken from Miles and Smith 2009
                        'Douglas-fir Bark': 440,
                        'Oak': 600,
                        'Pine': 400,
                        'Pine Bark': 350,
                        'Wheat Straw': 1038  # From Lam et al. 2008
                        }

solid_k_values = {'Douglas-fir': 0.12,  # FPL-GTR-190 4-7
                  'Douglas-fir Bark': 0.21,  # Gupta2003 table 3
                  'Oak': 0.15,  # FPL-GTR-190 4-7
                  'Pine': 0.10,  # FPL-GTR-190 4-7
                  'Pine Bark': 0.21,  # Gupta2003 table 3
                  'Wheat Straw': 0.16  # From Mason2016
                  }

solid_cp_values = {'Douglas-fir': 1.255,  # FPL-GTR-190 4-16a
                   'Douglas-fir Bark': 1.364,  # Gupta2003
                   'Oak': 1.225,  # FPL-GTR-190 4-16a
                   'Pine': 1.225,  # FPL-GTR-190 4-16a
                   'Pine Bark': 1.364,  # Gupta2003
                   'Wheat Straw': 1.340  # From Stenseng2001
                   }

cp_fluid = 1007
k_fluid = 26.3E-3
rho_fluid = 1
fourier_max = 0.075
v_container = 9.5E-4


if __name__ == '__main__':
    directories = ['/home/derek/Documents/IgnitionData/SpeciesData/']
    test_data = load_data(directories)
    test_data.loc[test_data['species'] == 'Oak Wood', 'species'] = 'Oak'
    mean_density_values = {groups[0]: groups[1]['density'].mean() for groups in test_data.groupby('species')}

    alpha_values = dict()
    for key, bulk_density in mean_density_values.items():
        rho_solid = solid_density_values[key]
        bed_density = bulk_density

        epsilon = 1 - np.divide(bed_density, rho_solid)
        k_eff_min = np.divide(1, (1 - epsilon) / solid_k_values[key] + epsilon / k_fluid)
        k_eff_max = epsilon * k_fluid + (1 - epsilon) * solid_k_values[key]
        # print(k_eff_min, k_eff_max, abs(k_eff_min-k_eff_max)/k_eff_min*100)
        k_bed = np.mean([k_eff_max, k_eff_min])
        # cp_bed = mean_density_values[f]/rho_solid*cp_solid
        cp_bed = epsilon * cp_fluid + (1 - epsilon) * solid_cp_values[key]
        alpha = np.divide(k_bed, bed_density * cp_bed)
        alpha_min = np.divide(k_eff_min, bed_density * cp_bed)
        alpha_max = np.divide(k_eff_max, bed_density * cp_bed)
        alpha_values[key] = [alpha_min, alpha, alpha_max]

