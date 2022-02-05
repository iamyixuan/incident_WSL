import numpy as np
import pandas as pd
import h5py
import argparse
import pickle

from preprocesing import fill_smooth_norm, filter_on_duration, filter_by_speed, filter_by_free_flow_speed
from utils import check_quality, create_lstm_data, gen_feat_eng_data, create_lstm_data_with_time_distance

"""
Load datasets:
D4: training and validation
D4_test: testing

Change datasets to march when generating testing set
"""

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2019)
parser.add_argument('--month', type=str, default='Jan')
args = parser.parse_args()

year = args.year
month = args.month

if year == 2019:
  profile_dir = '../data/processed_2019/'
  incident_dir = '../data/2019_accidents/'
  month_dir = args.month + '_concat/'
elif year == 2020:
  profile_dir = '../data/processed_2020_jan_feb_march/'
  incident_dir = '../data/processed_2020_jan_feb_march/'
  month_dir = ''
  



flow = pd.read_hdf(profile_dir + month_dir + '2019_04_flow.h5')
occ = pd.read_hdf(profile_dir + month_dir + '2019_04_occupancy.h5')
speed = pd.read_hdf(profile_dir + month_dir + '2019_04_speed.h5')

incident = pd.read_hdf(incident_dir + '2019_04_corrected.h5')
# print("original num incident", incident.shape[0])
#incident = pd.read_csv('../data/processed_2020_jan_feb_march/corrected_incident_train.csv')
norm_speed, norm_flow, norm_occ, speed, flow, occ = fill_smooth_norm(speed, flow, occ)


"""
we have (for D4):
* incidents in Jan and Feb. 2020 (March for testing)
* incidnt time, duration, upstream node ID and downstream node ID
* Speed, flow, occupancy for all these nodes.
we want:
* match the incidents with features.
* extract feature statistics.

* Need to make sure to use the same incidents
  for both baseline and lstm models.
"""
incident_filterd = filter_on_duration(filter_by_free_flow_speed(incident, norm_speed))
low_qual_idx = check_quality(incident_filterd, norm_speed, norm_flow, norm_occ)

#incident_filterd = pd.read_csv('../data/processed_2020_jan_feb_march/corrected_incident_test.csv')
#incident_filterd = pd.read_hdf(incident_dir + '2019_02_corrected.h5')
print(incident_filterd.shape)


#data = create_lstm_data(incident_filterd, norm_speed, norm_flow, norm_occ, low_qual_idx, 2019+12, 20)

#data_feat_eng =  gen_feat_eng_data(incident_filterd, norm_speed, norm_flow, norm_occ, low_qual_idx,  190*12, 10)

data = create_lstm_data_with_time_distance(incident_filterd, "../data/raw_accidents_i80_D4_2019/I80_D4_sensors.xlsx", norm_speed, norm_flow, norm_occ, low_qual_idx, 2019+4, 20)

print(data["timestamps"][0:10])
print(data["sensor_dist"][0:10])

# with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/2019_12.h5', 'w') as f:
#     f.create_dataset('input', data=np.array(data['input']))
#     f.create_dataset('output', data=np.array(data['output']))
#     f.create_dataset('event_num', data=np.array(data['event_num']))
#     f.create_dataset('profiles', data=np.array(data['profiles']))
#     f.create_dataset('timestamps', data=np.array(data['timestamps']))
#     f.create_dataset('sensor_dist', data=np.array(data['sensor_dist']))


with open('../data/lstm_norm_data/long_horizon/final/testTimeDistance/val/2019_04.pkl', 'wb') as handle:
  pickle.dump(data, handle)

# with h5py.File('../data/lstm_norm_data/long_horizon/final/2019_12.h5', 'w') as f:
#     f.create_dataset('input', data=np.array(data_feat_eng['input']))
#     f.create_dataset('output', data=np.array(data_feat_eng['output']))
#     f.create_dataset('event_num', data=np.array(data_feat_eng['event_num']))
