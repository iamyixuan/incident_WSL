import numpy as np
from numpy import diff
import pandas as pd
import h5py
from preprocesing import fill_smooth_norm

"""
Load datasets:
D4: training and validation
D4_test: testing

Change datasets to march when generating testing set
"""

# flow = pd.read_hdf('../data/processed_2019/Mar_concat/2019_03_flow.h5')
# occ = pd.read_hdf('../data/processed_2019/Mar_concat/2019_03_occupancy.h5')
# speed = pd.read_hdf('../data/processed_2019/Mar_concat/2019_03_speed.h5')

incident = pd.read_hdf('../data/2019_accidents/2019_03_corrected.h5')
# """
# forward fill the missing values
# """
# mask_speed = speed.values==0
# mask_flow = flow.values==0
# mask_occ = occ.values==0

# speed.iloc[mask_speed] = np.nan
# flow.iloc[mask_flow] = np.nan
# occ.iloc[mask_occ] = np.nan
# speed = speed.fillna(method='ffill', axis=0) # only forward fill speed
# flow = flow.fillna(method='ffill', axis=0)
# occ = occ.fillna(method='ffill', axis=0)
# speed = speed.interpolate(axis=1)
# flow = flow.interpolate(axis=1)
# occ = occ.interpolate(axis=1)

# """
# smoothening & normalization
# """

# speed = speed.ewm(span=5).mean()
# flow = flow.ewm(span=5).mean()
# occ = occ.ewm(span=5).mean()

# free_speed = speed.quantile(0.95)
# free_flow = flow.quantile(0.95)
# free_occ = occ.quantile(0.95)

# norm_speed = speed/free_speed
# norm_flow = flow/free_flow
# norm_occ = occ/free_occ
speed = pd.read_hdf('../data/processed_2019/Mar_concat/2019_03_speed.h5')
flow = pd.read_hdf('../data/processed_2019/Mar_concat/2019_03_flow.h5')
occ = pd.read_hdf('../data/processed_2019/Mar_concat/2019_03_occupancy.h5')


norm_speed, norm_flow, norm_occ = fill_smooth_norm(speed, flow, occ)

def get_seqs(incident, speed, flow, occ, start_idx, end_idx):
    seqs = []
    for i in range(start_idx, end_idx):
        start_t = pd.to_datetime(incident.iloc[i]['Time'])
        duration = incident.iloc[i]['Duration']
        up = incident.iloc[i]['Upstream ID']
        down = incident.iloc[i]['Downstream ID']
        if duration <0:
            end_t = start_t + pd.Timedelta(20*60, unit='s') # mark the incident span we wanna use
        else:
            end_t = start_t + pd.Timedelta(duration*60, unit='s')
        sample_start = start_t - pd.Timedelta(1*3600, unit='s')
        sample_end = start_t + pd.Timedelta(2*3600, unit='s')
        speed_seq = speed.loc[sample_start:sample_end, up]
        flow_seq = flow.loc[sample_start:sample_end, up]
        occ_seq = occ.loc[sample_start:sample_end, up]
        seqs.append([speed_seq, flow_seq, occ_seq])
    return seqs

"""
calculate the first order derivate seqs
and get the mean and std.

get the mean of the mean values in each diff seqs
get the std of the mean values in each diff seqs

if the new sequences have much lower (exeeding one std) first order deritive it might be a low quality data point
"""
# s = get_seqs(incident, norm_speed, norm_flow, norm_occ, 0, len(incident))


# ddt = np.abs(diff(s, axis=2)) # should take the absolute values of the changes
# mean_diff = np.mean(ddt, axis=(0, 2)) # get the mean of mean
# std_mean_diff = np.std(np.mean(ddt, axis=2), axis=0)
# print(mean_diff, std_mean_diff)

"""
diff mean of mean= [0.01566702 0.03229781 0.04392279] 
diff std of mean = [0.01043915 0.02578795 0.04089887]

if the new data point comes in with profile
"""


def check_quality(incident, speed, flow, occ, start_idx, end_idx):
    for i in range(start_idx, end_idx):
        start_t = pd.to_datetime(incident.iloc[i]['Time'])
        duration = incident.iloc[i]['Duration']
        up = incident.iloc[i]['Upstream ID']
        down = incident.iloc[i]['Downstream ID']
        if duration <0:
            end_t = start_t + pd.Timedelta(20*60, unit='s') # mark the incident span we wanna use
        else:
            end_t = start_t + pd.Timedelta(duration*60, unit='s')
        sample_start = start_t - pd.Timedelta(1*3600, unit='s')
        sample_end = start_t + pd.Timedelta(2*3600, unit='s')
        speed_seq = speed.loc[sample_start:sample_end, up]
        flow_seq = flow.loc[sample_start:sample_end, up]
        occ_seq = occ.loc[sample_start:sample_end, up]
        ddt_speed = np.abs(diff(speed_seq)).mean()
        ddt_flow = np.abs(diff(flow_seq)).mean()
        ddt_occ = np.abs(diff(occ_seq)).mean()
        if i == 24:
            print(ddt_speed, ddt_flow, ddt_occ)
        if np.mean(ddt_speed) <= (0.01566702 - 0.01043915) or np.mean(ddt_flow) <= (0.03229781 - 0.02578795) or np.mean(ddt_occ) <= (0.04392279 - 0.04089887):
            print("incident with ID {} might be low quality data".format(i))
    return 0


check_quality(incident, norm_speed, norm_flow, norm_occ, 0, len(incident))