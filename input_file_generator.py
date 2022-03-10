import json
import os
import pandas as pd
from datetime import datetime


def create_input_file(filepath, **kwargs):
    '''
     The input file for the TestData class that is contained in the ignitionTestAnalysis file.
     The input file is a json file that contains a dictionary with information necessary for
     data analysis. The format of the file is a two level dictionary in the following format:
         Fuel Morphology
             -species
             -moisture content
             -particle size
             -typical morphology
         Test Parameters
             -setpoint
             -container volume
             -sample mass
             -date
             -notes
         File Information
             -tdms file
             -output file
             -save data
     '''
    function_inputs = {'species': "Douglas-fir",
                       'moisture_content': 10,
                       'particle_size': None,
                       'typical_morphology': None,
                       'setpoint': None,
                       'wind_speed':None,
                       'container_volume': 950,
                       'sample_mass': None,
                       'date': None,
                       'notes': None,
                       'tdms_file': filepath,
                       'output_file': filepath.split('.')[0] + '.json',
                       'save_data': False,
                       'apparatus': 0,
                       'ignition': True}

    for key, value in kwargs.items():
        function_inputs[key] = value

    # generate the double layer dictionary with the keys specified in the docstring
    first_layer_keys = ['fuel_morphology', "test_parameters", "file_information"]

    second_layer_keys = [['species', 'moisture_content', 'particle_size', 'typical_morphology'],
                         ['setpoint', 'container_volume', 'sample_mass', 'date', 'notes', 'ignition', 'apparatus', 'wind_speed'],
                         ['tdms_file', 'output_file', 'save_data']]

    input_dict = {key: dict.fromkeys(second_layer_keys[idx], None) for idx, key in enumerate(first_layer_keys)}

    input_dict['fuel_morphology'].update(species=function_inputs['species'],
                                         moisture_content=function_inputs['moisture_content'],
                                         particle_size=function_inputs['particle_size'],
                                         typical_morphology=function_inputs['typical_morphology'])

    input_dict['test_parameters'].update(setpoint=function_inputs['setpoint'],
                                         container_volume=function_inputs['container_volume'],
                                         sample_mass=function_inputs['sample_mass'],
                                         date=function_inputs['date'],
                                         notes=function_inputs['notes'],
                                         wind_speed=function_inputs['wind_speed'],
                                         apparatus=function_inputs['apparatus'],
                                         ignition=function_inputs['ignition'])

    input_dict['file_information'].update(tdms_file=function_inputs['tdms_file'],
                                          output_file=function_inputs['output_file'],
                                          save_data=function_inputs['save_data'])
    os.makedirs(os.path.dirname(filepath.split('.')[0]+'_input.json'), exist_ok=True)
    with open(filepath.split('.')[0]+'_input.json', 'w+') as f:
        json.dump(input_dict, f, indent=1)
    return input_dict

def find_data_files(directory_paths):
    filenames = []
    for directory in directory_paths:
        for _, _, files in os.walk(directory):
            for filename in files:
                if '.tdms' in filename.lower() and 'index' not in filename.lower() and 'test' in filename.lower():
                    filenames.append(os.path.join(directory, filename))
    return filenames



if __name__ == "__main__":
    # log_file = 'C:\\Users\\derek\\Desktop\\IgnitionData\\CurrentData\\ign_test_logs.csv'
    log_file = '/home/derek/Documents/IgnitionData/SpeciesData/species_data.csv'
    log_data = pd.read_csv(log_file)
    log_data['date'] = log_data['file_number'].str.split('_').str[1].str.split('.').str[0]
    log_data['date'] = pd.to_datetime(log_data['date'], format='%m%d%Y')
    log_data['folder'] = 'Ignition_' + log_data['date'].dt.strftime("%m_%d_%Y")
    # data_path = 'C:\\Users\\derek\\Desktop\\IgnitionData\\CurrentData\\'
    data_path = '/home/derek/Documents/IgnitionData/SpeciesData/'

    log_data['input_filepath'] = data_path + log_data['folder'] + os.sep + log_data['file_number']
    log_data['input_filepath'] = log_data['input_filepath'].str.split('.').apply(lambda x: x[0] + '_input.json')
    for idx, j in log_data.iterrows():
        create_input_file(os.path.join(data_path, j.folder, j.file_number),
                          sample_mass=j.mass,
                          setpoint=j.temperature,
                          output_file='_'.join(j.input_filepath.split('_')[:-1])+'_output.json',
                          particle_size='0.85mm to 2.12mm',
                          species=j.material,
                          date=j.date.strftime("%m/%d/%Y"),
                          ignition=j.ignition,
                          apparatus=0,
                          wind_speed=j.wind_speed,
                          moisture_content=0)

    # for j in files_to_create.index.values:
    #     create_input_file(files_to_create['files'][j] + '.tdms',
    #                       sample_mass=files_to_create['sample_mass'][j],
    #                       setpoint=int(files_to_create['setpoint'][j]),
    #                       particle_size=files_to_create['particle_size'][j],
    #                       date=files_to_create['test_date'][j],
    #                       species=files_to_create['species'][j],
    #                       ignition=bool(files_to_create['ignition'][j]),
    #                       apparatus=int(files_to_create['apparatus'][j])
    #                       )
    # # for j in log_data.index.values:
    # #     create_input_file(log_data['files'][j] + '.tdms',
    # #                       sample_mass=log_data['sample_mass'][j],
    # #                       setpoint=int(log_data['setpoint'][j]),
    # #                       particle_size=log_data['particle_size'][j],
    # #                       date=log_data['test_date'][j],
    # #                       species=log_data['species'][j],
    # #                       ignition=bool(log_data['ignition'][j]),
    # #                       apparatus=int(log_data['apparatus'][j])
    # #                       )
    #
