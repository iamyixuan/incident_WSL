import numpy as np
import pandas as pd
import os
import h5py

from preprocesing import aggregate_raw, concat_info_xlxs, concat_info_h5, match_incidents

"""
Processing raw traffic profiles into days
"""
# sensor_list = pd.read_excel('../data/raw_accidents_i80_D4_2019/I80_D4_sensors.xlsx')
# aggregate_raw('../data/raw_station/12_Dec/', '../data/processed_2019/', list(sensor_list['ID']))

"""
Aggregate and match incidents based on the senesors of interest
"""
# incident_dir = '../data/raw_accidents_i80_D4_2019/Dec/'
# target_dir = '../data/2019_accidents/'

# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)

# concat_info_xlxs(incident_dir, target_dir, '2019_12_accidents')

"""
Match incidents with the correponding sensors
"""
# incident_matched = match_incidents('../data/raw_accidents_i80_D4_2019/I80_D4_sensors.xlsx', '../data/2019_accidents/2019_12_accidents.h5')
# df_sorted  = incident_matched.sort_values(by=['Time'])
# df_sorted.to_hdf('../data/2019_accidents/2019_12_matched.h5', key='df')
# print(df_sorted.shape)


# """
# Aggregate traffic profiles

# speed, flow, and occupancy
# """
# profiles = ['speed', 'flow', 'occupancy']

# target_profile_dir = '../data/processed_2019/Dec_concat/'

# if not os.path.exists(target_profile_dir):
#     os.makedirs(target_profile_dir)

# for prof_name in profiles:
#     profile_dir = '../data/processed_2019/Dec/' + prof_name + '/'


#     concat_info_h5(profile_dir, target_profile_dir, '2019_12_' + prof_name)

"""
concatenate processed lstm data
"""
def concat_lstm_data(f_path, filename_list):
    for i, f_name in enumerate(filename_list):
        f = h5py.File(f_path + f_name, 'r')
        if i == 0:
            x = f['input'][...]
            y = f['output'][...]
            z = f['event_num'][...]
            m = f['profiles'][...]
        else:
            x = np.concatenate((x, f['input'][...]), axis=0)
            y = np.concatenate((y, f['output'][...]), axis=0)
            z = np.concatenate((z, f['event_num'][...]), axis=0)
            m = np.concatenate((m, f['profiles'][...]), axis=0)

    return x, y, z, m


dir_name = '../data/lstm_norm_data/free_flow_filtered/test/'

f_list = os.listdir(dir_name)
print(f_list)

if '.DS_Store' in f_list:
    f_list.remove('.DS_Store')

x, y, z, m = concat_lstm_data(dir_name, f_list)
print(x.shape, y.shape)
with h5py.File(dir_name + 'test_with_profile.h5', 'w') as f:
    f.create_dataset('input', data=x)
    f.create_dataset('output', data=y)
    f.create_dataset('event_num', data=z)
    f.create_dataset('profile', data=m)


