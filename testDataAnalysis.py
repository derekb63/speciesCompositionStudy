# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:44:02 2019

@author: beande
"""
import pandas as pd
from nptdms import TdmsFile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import os
import re
import scipy.fftpack as fftpack
import sys
import mplcursors


class TestData:
    """
    A collection of tools used to analyze data collected in tests for the ignition project.
    """

    def __init__(self, input_file):
        """
        Initialize the TestData object by loading some of the most important details

        :param input_file: a string that states the location of the input file for the test

        :return None:

        The variables that are initialized are:
        dataFile: The location of the .tdms file that contains the data that is collected
        saveFile: The location of the .json file where the data will be saved
        _heaterTime: The point in time where the heater is dropped onto the fuel bed
        _ignitionTime: The point in time where the fuel bed is ignited by the heater.
        """
        with open(input_file, 'r', encoding='utf-8') as file:
            self.inputs = json.load(file)

        self.ignitionEvent = self.inputs['test_parameters']['ignition']
        self.dataFile = self.inputs['file_information']['tdms_file']
        self.saveFile = self.inputs['file_information']['output_file']
        self.saveData = self.inputs['file_information']['save_data']

        with open(self.dataFile, 'rb') as datafile:
            self.data = TdmsFile(datafile)
        self.parse_data()
        self.set_heater_time()
        self.set_ignition_time()
        self._heaterTime = self.photodiode['time'][0]
        self._ignitionTime = self.photodiode['time'][-1]

    def parse_data(self):
        """
        Separate the data into dictionaries of data and timestapmps
        """
        test_groups = self.data.groups()
        if len(test_groups) > 1:
            warnings.warn("There is more than one dataset in this file.\
                           Only the first dataset will be imported")
        # TODO: Add in some sort of tool to handle multiple tests in one file
        self.dataDict = {}
        for group in test_groups:
            for channel in group.channels():
                self.dataDict[channel.name] = self.data[group.name][channel.name]
        # self.dataDict = {dataset.channel: dataset for dataset in self.data.group_channels(test_groups[0])}
        # self.dataDict = self.data.as_dataframe()
        # self.dataDict.columns = channels
        # self.dataDict = self.dataDict.to_dict()
        return self.dataDict

    @staticmethod
    def find_nearest_datapoint(data_list, value):
        '''
        Given a list and a value find the index of the closest list item to the value
        :param data_list:
        :param value:
        :return:
        '''
        return np.where(data_list == min(data_list, key=lambda x: abs(x - value)))

    @staticmethod
    def moisture_content(dry_mass, wet_mass):
        return

    @property
    def photodiode(self):
        try:
            return {'data': self.dataDict['Photodiode'].data,
                    'time': self.dataDict['Photodiode'].time_track()}
        except KeyError:
            print(self.dataFile)

    @property
    def temperature(self):
        return {'data': self.dataDict['Temperature'].data,
                'time': self.dataDict['Temperature'].time_track()}

    @property
    def power(self, calc_method='recorded'):
        if calc_method == 'recorded':
            return {'data': self.dataDict['Power'].data,
                    'time': self.dataDict['Power'].time_track()}
        elif calc_method == 'calculated':
            return {'data': np.multiply(self.voltage['data'], self.current['data']),
                    'time': self.voltage['time']}

    @property
    def voltage(self):
        return {'data': self.dataDict['Voltage'].data,
                'time': self.dataDict['Voltage'].time_track()}

    @property
    def voltage_calibration(self):
        '''
        Apply the voltage calibration curve to the recorded
        data to get the AC voltage supplied to the heater
        :return:  AC voltage supplied to the heater
        '''
        m_1 = 0.016
        m_2 = 0.511
        b = 2.59
        return np.multiply(m_1, np.power(self.voltage,2)) + np.multiply(m_2, self.voltage) + b

    @property
    def voltage_error(self):
        return None

    @property
    def current(self):
        return {'data': self.dataDict['Current'].data,
                'time': self.dataDict['Current'].time_track()}

    @property
    def current_calibration(self):
        m = 0.526
        b = -0.053
        return np.multiply(m, self.current) + b

    def set_ignition_time(self):
        self._ignitionTime = self.temperature['time'][self.temperature['data'].argmax()]
        # self._ignitionTime = self.photodiode['time'][-1]
        # if self.ignitionEvent:
        #     self._ignitionTime = data.photodiode['time'][data.find_nearest_datapoint(data.photodiode['time'], value)]
        # else:
        #     self._ignitionTime = data.photodiode['time'][-1]

    def get_ignition_time(self):
        return self._ignitionTime

    ignition_time = property(fget=get_ignition_time, fset=set_ignition_time)

    def set_heater_time(self):
        self._heaterTime = self.photodiode['time'][0]
        # self._heaterTime = float(data.photodiode['time'][data.find_nearest_datapoint(data.photodiode['time'], value)])

    def get_heater_time(self):
        return self._heaterTime

    heater_time = property(fset=set_heater_time, fget=get_heater_time)

    def get_test_duration(self):
        return float(self.ignition_time - self.heater_time)

    def set_test_duration(self, value):
        self.test_duration = value

    test_duration = property(fset=set_test_duration, fget=get_test_duration)

    @property
    def mean_temperature(self):
        return self.temperature['data'].mean()
        # return self.temperature['data'][self.find_nearest_datapoint(data.temperature['time'], self.ignition_time)]

    def mechanical_switch_finder(self):
        '''
        Detect the changes in state in the limit switch
        :return: The location of the peaks
        '''

    def photodiode_peak_finder(self):
        # threshold = 1.0
        finder_data = self.derivative_photodiode()['data']
        threshold = len(finder_data)
        peaks = signal.find_peaks(finder_data, distance=threshold)
        while len(peaks[0]) < 2:
            threshold -= threshold * 0.01
            try:
                peaks = signal.find_peaks(finder_data, distance=threshold)
            except ValueError:
                peaks = [[0, 0],]
        return peaks[0][0], peaks[0][-1]

    def mean_power(self):
        return np.mean(self.power['data'][self.find_nearest_datapoint(self.power['time'], self.heater_time)[0][0]:
                                         self.find_nearest_datapoint(self.power['time'], self.ignition_time)[0][0]])

    def energy_error(self):
        # TODO: complete the energy error determination function
        np.sqrt(
            np.power(
                np.multiply(
                    self.current_calibration, self.voltage_error
                )
            , 2)
            +
            np.power(
                np.multiply(
                    self.voltage_calibration, self.current_error
                )
            , 2)
        )

    def filter_photodiode(self, n=3, wn=0.005):
        '''
        Apply a butterworth filter to the photodiode data to eliminate some of the high
        frequency noise that results from the nearby AC components and other noise sources
        :param n:  The order of the filter. 3 was chosen as a default since it returns a reasonable result
        :param wn: The critical frequency of the filter
        :param fs: The sampling frequency of the system
        :return: a dictionary containing the filtered data and the corresponding time values
        '''
        b, a = signal.butter(n, wn, output='ba', btype='lowpass')
        filter_output = signal.filtfilt(b, a, self.photodiode['data'])
        return {'data': filter_output, 'time': self.photodiode['time']}

    def derivative_photodiode(self):
        '''
        Return the derivative of the photodiode signal. This is a first order difference of
        the photodiode signal
        :return: a dictionary containing the differenced data and the corresponding time values
        '''
        return {'data': np.diff(self.filter_photodiode()['data']),
                'time': self.photodiode['time'][:-1]}

    @property
    def density(self):
        mass = self.inputs['test_parameters']['sample_mass']*1E-3
        volume = self.inputs['test_parameters']['container_volume']*1E-6
        return mass / volume

    def save_data(self):
        """
        creates a dictonary that contains all of the data and parameters for the tests
        The dictionary is structured in the following manner
        - Fuel morphology
            - species
            - moisture content
            - particle size
            - typical morphology
        - Test parameters
            - density
            - setpoint
            - container volume
            - sample mass
            - date
            - apparatus
            - notes
        - Data
            - temperature
            - photodiode
            - voltage
            - current
            - power
        -Results
            - ignition
            - time to ignition
            - mean_temperature
            - total energy
            - mean_power
            - lower_power
            - heater_time
        :return:
        """

        first_level_keys = ['fuel_morphology', 'test_parameters', 'results']

        second_level_keys = [['species', 'moisture_content', 'particle_size', 'typical_morphology'],
                             ['density', 'setpoint', 'container_volume', 'sample_mass', 'date', 'apparatus', 'notes',
                              'ignition', 'wind_speed', 'heater_lead_angle'],
                             ['time_to_ignition', 'mean_temperature', 'mean_power']]

        data_dict = {key: dict.fromkeys(second_level_keys[idx], None) for idx, key in enumerate(first_level_keys)}

        data_dict['test_parameters'].update(density=self.density)

        # if self.saveData:
        #     data_dict['data'].update(temperature=jsonify(self.temperature),
        #                              photodiode=jsonify(self.photodiode),
        #                              voltage=jsonify(self.voltage),
        #                              current=jsonify(self.current),
        #                              power=jsonify(self.power))
        #
        data_dict['results'].update(mean_power=self.mean_power(),
                                    mean_temperature=self.mean_temperature)
        if not self.ignitionEvent:
            data_dict['results'].update(time_to_ignition=None)
        else:
            data_dict['results'].update(time_to_ignition=self.test_duration)

        for idx, firstKey in enumerate(first_level_keys):
            for secondKey in second_level_keys[idx]:
                try:
                    data_dict[firstKey][secondKey] = self.inputs[firstKey][secondKey]
                except (AttributeError, KeyError):
                    warnings.warn('There are some empty data slots')
                    # TODO: add in a section to fill in the ones that fail
        with open(self.saveFile, 'w+') as f:
            json.dump(data_dict, f, indent=1)


def find_inputs(directory_paths):
    filenames = []
    for directory in directory_paths:
        for folder, _, files in os.walk(directory):
            for filename in files:
                if 'input' in filename.lower() and '.json' in filename.lower():
                    filenames.append(os.path.join(directory, os.path.join(folder, filename)))
    return filenames


if __name__ == "__main__":
    # directories = ['C:/Users\\derek\\Desktop\\IgnitionData\\WindTunnelData\\Ignition_09_30_2021']
    #                'C:\\Users\\derek\\Desktop\\IgnitionData\\CurrentData\\Ignition_10_15_2020']
    directories = ['/home/derek/Documents/IgnitionData/SpeciesData']
    fNames = [x for x in find_inputs(directories) if 'test' in x.lower()]
    # photoDiodeData = pd.DataFrame(index=np.arange(0, 3000, 0.001))
    # for fileInput in fNames:
    #     try:
    #         test_name = '_'.join(fileInput.split('/')[-1].split('_')[:2])
    #         photoDiodeData[test_name] = pd.DataFrame.from_dict(TestData(fileInput).photodiode).set_index('time')
    #     except:
    #         print('Error in: ' + fileInput)
    temperature_data = []
    photodiode_data = []
    peak_values = {}
    window_steps = np.linspace(0, 1, 101)
    threshold_points = pd.DataFrame(index=window_steps)
    for inputFile in fNames:
        if os.path.isfile(inputFile.replace('input', 'output')):
            pass
        else:
            data = TestData(inputFile)
            print(data.inputs['file_information']['tdms_file'])
            data_column = "_".join(re.split('_|/', inputFile)[-3:-1])
            test_data = pd.Series(data.photodiode['data'], name=data_column)
            # plt.figure()
            # test_data.plot(logx=True, title=data_column)
            # test_data[test_data > 0.6 * test_data.max()].plot(logx=True, title=data_column)
            photodiode_data.append(test_data)
            # peaks[data_column] = signal.find_peaks(pd.Series(data.photodiode['data']).diff(), threshold=0.25)[0]
            # for i in window_steps:
            #     threshold_points.loc[i, data_column] = test_data[test_data > i].shape[0]

            # if data.ignitionEvent:
            #     data.check_duration()
            # else:
            #     data.check_heater_time()
            data.save_data()
    # photodiode_data = pd.concat(photodiode_data, axis=1)
    # # threshold_limits = threshold_points.diff().idxmin()
    # time_offset = 2500
    # for col_name, col_data in photodiode_data.iteritems():
    #     max_value = col_data.diff().max()
    #     peak_threshold = 1
    #     distance = 500
    #     peaks = signal.find_peaks(col_data.iloc[time_offset:].diff(), height=max_value * peak_threshold, distance=1)
    #     while len(peaks[0]) > 1:
    #         peak_threshold -= peak_threshold * 0.01
    #         try:
    #             peaks = signal.find_peaks(col_data.iloc[time_offset:].diff(), height=max_value * peak_threshold, distance=1)
    #         except ValueError:
    #             peaks = [[0, 0], ]
    #     peak_values[col_name] = peaks
    # peak_values = pd.DataFrame.from_dict(peak_values, orient='index')[0]
    # # import seaborn as sns
    # # sns.histplot(peak_values, kde=True, log_scale=True)
    # # for test_name, lower_point in peak_values.iteritems():
    # #     plt.figure()
    # #     photodiode_data[test_name].diff().plot()
    # #     photodiode_data[test_name].plot(title=test_name)
    # #     plt.scatter(lower_point, photodiode_data[test_name][lower_point], marker='x', c='r')
