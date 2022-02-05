import os
import numpy as np
from numpy import diff
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import label
#from feature_eng import FeatureExtractor

def get_norm_seq(incident, speed, flow, low_qual_idx, horizon):
    """
    Get normal traffic profile for anomaly detection using 
    convolutional autoencoder
    """
    data = {"norm_seqs": [], "abnormal_seqs":[]}
    for i in range(incident.shape[0]):
        if i in low_qual_idx:
            pass 
        else:
            up_sensor = incident.iloc[i]["Upstream ID"]
            down_sensor = incident.iloc[i]["Downstream ID"]
            incident_start_t = pd.Timedelta(incident.iloc[i]["Time"])
            incident_end_t = incident_start_t + pd.Timedelta(incident[i]["Duration"]*60, unit='s')
            incidnet_span = pd.date_range(incident_start_t, incident_end_t, freq='30s')








def create_lstm_data(incident, speed, flow, occ, low_qual_idx, month_num, horizon):
    """
    create data for the LSTM model
    three + one features: speed, flow, occupancy and seq after convolution
    Returns a dictionary with input and output keys.

    incident: cleaned incident Dataframe.
    speed: processed speed Dataframe.
    flow: processed volume/flow Dataframe.
    occ: processed occupancy Dataframe.
    start_idx: starting index for the incident Dataframe.
    end_idx: ending index for the incident Dataframe.

    return a dictionary containing input and output pairs
    input shape: (num_points, 6, 5)
    output shape: (num_points, 1)
    """
    data = {'input':[], 'output':[], 'event_num':[], 'profiles': []}
    for i in range(incident.shape[0]):
        if i in low_qual_idx:
            print("Took out {} example for it's low quality".format(i))
            pass
        else:
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
            incident_span = pd.date_range(start_t, end_t, freq='30s')
            sample_span = pd.date_range(sample_start, sample_end, freq='30s') # change the end time based on trianing/validation
            time_ahead = pd.Timedelta(horizon*60, unit='s') # history length


            if (int(up) in speed.columns) and (int(down) in speed.columns):

                for t in sample_span:
                    up_speed = speed.loc[t-time_ahead:t, up].values
                    up_vol = flow.loc[t-time_ahead:t, up].values
                    up_occ = occ.loc[t-time_ahead:t, up].values

                    down_speed = speed.loc[t-time_ahead:t, down].values
                    down_vol = flow.loc[t-time_ahead:t, down].values
                    down_occ = occ.loc[t-time_ahead:t, down].values
                    # Here add the extra feature
                    if len(up_speed) == (horizon*60/30 + 1):
                        up_down_conv = np.convolve(up_speed, down_speed, 'same')
                        speed_u = speed.loc[sample_start : sample_end, up].values
                        speed_d = speed.loc[sample_start : sample_end, down].values

                        vol_u = flow.loc[sample_start : sample_end, up].values
                        vol_d = flow.loc[sample_start : sample_end, down].values

                        occ_u = occ.loc[sample_start : sample_end, up].values
                        occ_d = occ.loc[sample_start : sample_end, down].values
                        
                        if t in incident_span and len(up_speed == 5):
                            data['input'].append([up_speed, up_vol, up_occ, down_speed, down_vol, down_occ, up_down_conv])
                            data['output'].append(1)
                            data['event_num'].append(0.5*(i + month_num + 1) * (i + month_num) + month_num)
                        else:
                            data['input'].append([up_speed, up_vol, up_occ, down_speed, down_vol, down_occ, up_down_conv])
                            data['output'].append(0)
                            data['event_num'].append(0.5*(i + month_num + 1) * (i + month_num) + month_num) # use a pairing function
                    else:
                        pass
                print(i)
                print(speed_u.shape, occ_u.shape, vol_u.shape, speed_d.shape, occ_d.shape, vol_d.shape)
                profile = np.array([speed_u, speed_d, vol_u, vol_d, occ_u, occ_d])
                if profile.shape != (6, 361):
                    tmp = np.zeros((6, 361 - profile.shape[1]))
                    profile = np.concatenate([profile, tmp], axis=1)
                data['profiles'].append(profile)
            else:
                pass
    return data

def create_lstm_data_with_time_distance(incident, sensorFile, speed, flow, occ, low_qual_idx, month_num, horizon):
    data = {'input':[], 'output':[], 'event_num':[], 'timestamps': [], 'sensor_dist': []}
    for i in range(incident.shape[0]):
        up = incident.iloc[i]['Upstream ID']
        down = incident.iloc[i]['Downstream ID']
        sensors = pd.read_excel(sensorFile)
        up_pm = sensors[sensors["ID"] == up]["Abs PM"].values
        down_pm = sensors[sensors["ID"] == down]["Abs PM"].values
       
        if i in low_qual_idx:
            print("Took out {} example for it's low quality".format(i))
            pass

        elif len(up_pm) < 1 or len(down_pm) < 1:
            print("No such sensor in list!")
            pass
        else:
            start_t = pd.to_datetime(incident.iloc[i]['Time'])
            duration = incident.iloc[i]['Duration']

            sensor_dist = float(down_pm[0] - up_pm[0])
            if duration <0:
                end_t = start_t + pd.Timedelta(20*60, unit='s') # mark the incident span we wanna use
            else:
                end_t = start_t + pd.Timedelta(duration*60, unit='s')
            sample_start = start_t - pd.Timedelta(1*3600, unit='s')
            sample_end = start_t + pd.Timedelta(2*3600, unit='s')
            incident_span = pd.date_range(start_t, end_t, freq='30s')
            sample_span = pd.date_range(sample_start, sample_end, freq='30s') # change the end time based on trianing/validation
            time_ahead = pd.Timedelta(horizon*60, unit='s') # history length


            if (int(up) in speed.columns) and (int(down) in speed.columns):

                for t in sample_span:
                    up_speed = speed.loc[t-time_ahead:t, up].values
                    up_vol = flow.loc[t-time_ahead:t, up].values
                    up_occ = occ.loc[t-time_ahead:t, up].values

                    down_speed = speed.loc[t-time_ahead:t, down].values
                    down_vol = flow.loc[t-time_ahead:t, down].values
                    down_occ = occ.loc[t-time_ahead:t, down].values
                    # Here add the extra feature
                    if len(up_speed) == (horizon*60/30 + 1):
                        up_down_conv = np.convolve(up_speed, down_speed, 'same')
                        speed_u = speed.loc[sample_start : sample_end, up].values
                        speed_d = speed.loc[sample_start : sample_end, down].values

                        vol_u = flow.loc[sample_start : sample_end, up].values
                        vol_d = flow.loc[sample_start : sample_end, down].values

                        occ_u = occ.loc[sample_start : sample_end, up].values
                        occ_d = occ.loc[sample_start : sample_end, down].values
                        
                        if t in incident_span and len(up_speed == 5):
                            data['input'].append([up_speed, up_vol, up_occ, down_speed, down_vol, down_occ, up_down_conv])
                            data['output'].append(1)
                            data['event_num'].append(0.5*(i + month_num + 1) * (i + month_num) + month_num)
                            data['timestamps'].append(t)
                            data['sensor_dist'].append(sensor_dist)
                        else:
                            data['input'].append([up_speed, up_vol, up_occ, down_speed, down_vol, down_occ, up_down_conv])
                            data['output'].append(0)
                            data['event_num'].append(0.5*(i + month_num + 1) * (i + month_num) + month_num) # use a pairing function
                            data['timestamps'].append(t)
                            data['sensor_dist'].append(sensor_dist)
                    else:
                        pass
                print(i)
                # profile = np.array([speed_u, speed_d, vol_u, vol_d, occ_u, occ_d])
                # if profile.shape != (6, 361):
                #     tmp = np.zeros((6, 361 - profile.shape[1]))
                #     profile = np.concatenate([profile, tmp], axis=1)
                # data['profiles'].append(profile)
            else:
                pass
    return data


def gen_feat_eng_data(incident, speed, flow, occ, low_qual_idx, month_num):
    data = {'input':[], 'output':[], 'event_num':[]}
    for i in range(incident.shape[0]):
        if i in low_qual_idx:
            print("Took out {} example for it's low quality".format(i))
            pass
        elif '08-31-19' in incident.iloc[i]['Time']:
            pass
        else:
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
            incident_span = pd.date_range(start_t, end_t, freq='30s')
            sample_span = pd.date_range(sample_start, sample_end, freq='30s') # change the end time based on trianing/validation

            speed_use = speed.loc[sample_start - pd.Timedelta(10, unit='m'): sample_end +  pd.Timedelta(10, unit='m')]
            flow_use = flow.loc[sample_start - pd.Timedelta(10, unit='m'): sample_end +  pd.Timedelta(10, unit='m')]
            occ_use = occ.loc[sample_start - pd.Timedelta(10, unit='m'): sample_end +  pd.Timedelta(10, unit='m')]

            for t in sample_span:
                if t in incident_span:
                    extractor = FeatureExtractor(speed_use, flow_use, occ_use, t, up, down, 1)
                else:
                    extractor = FeatureExtractor(speed_use, flow_use, occ_use, t, up, down, 0)
                feat, label = extractor.get_features()
                data['input'].append(feat)
                data['output'].append(label)
                data['event_num'].append(0.5*(i + month_num + 1) * (i + month_num) + month_num)
    return data







def viz_incidents(speed, flow, occ, incident, low_qual_idx, dir_name):
    """
    visualize all the incidents in the incident Dataframe

    speed: speed Dataframe (either normalized or original)
    flow: volume Dataframe (either normalized or original)
    occ: occupancy Dataframe (either normalized or original)
    incident: incident Dataframe.
    """
    mpl.style.use('seaborn-deep')
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', "Sun"]
    sensors = speed.columns
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i in range(incident.shape[0]):
        if i in low_qual_idx:
            print("Took out {} example for it's low quality".format(i))
            pass
        else:
            date = pd.to_datetime(incident.iloc[i]['Time']) - pd.Timedelta(3600, unit='s') 
            times = pd.to_datetime(incident.iloc[i]['Time'])
            # if incident.iloc[i]['Duration'] > 2*60:
            #     date2 = times + pd.Timedelta(incident.iloc[i]['Duration']*60, unit='s') 
            # else:
            date2 = times + pd.Timedelta(2*3600, unit='s')
            num = i    
            duration = incident.iloc[num]['Duration']*60
            upstr_node_pos = incident.iloc[num]['Upstream ID']
            downstr_node_pos = incident.iloc[num]['Downstream ID']
            fig, ax = plt.subplots(1, 3, figsize=(21, 3))
            ax[0].plot(range(speed.loc[date:date2].shape[0]), speed.loc[date:date2, upstr_node_pos], label='upstream')
            ax[0].plot(range(speed.loc[date:date2].shape[0]), speed.loc[date:date2, downstr_node_pos], label='downstream')
            ax[0].set_xlabel('Timestamps')
            if duration <= 2*3600:
                incident_end_time = times + pd.Timedelta(duration, unit='s')
            else:
                incident_end_time = date2
            try:
                ax[0].axvspan(times, incident_end_time, color='red', alpha=0.1)
            except:
                pass
            ax[0].set_ylabel('speed')
            ax[0].grid('off')
            ax[0]
            ax[0].legend()

            ax[1].plot(range(speed.loc[date:date2].shape[0]), flow.loc[date:date2, upstr_node_pos], label='upstream')
            ax[1].plot(range(speed.loc[date:date2].shape[0]), flow.loc[date:date2, downstr_node_pos], label='downstream')
            ax[1].set_xlabel('Timestamps')
            try:
                ax[1].axvspan(times, incident_end_time,  color='red', alpha=0.1) #times + pd.Timedelta(duration, unit='s'),
            except:
                pass
            ax[1].set_ylabel('flow')
            ax[1].grid('off')
            ax[1].legend()

            ax[2].plot(range(speed.loc[date:date2].shape[0]), occ.loc[date:date2, upstr_node_pos], label='upstream')
            ax[2].plot(range(speed.loc[date:date2].shape[0]), occ.loc[date:date2, downstr_node_pos], label='downstream')
            ax[2].set_xlabel('Timestamps')
            try:
                ax[2].axvspan(times, incident_end_time, color='red', alpha=0.1)
            except:
                pass
            ax[2].grid('off')
            ax[2].set_ylabel('occupancy')
            ax[2].legend()
            day_of_the_week = days[int(pd.Timestamp(times).dayofweek)]
            hour_of_the_day = str(pd.Timestamp(times).hour)
            minute_of_the_day = str(pd.Timestamp(times).minute)
            #fig.suptitle("Weekday: " + day_of_the_week + "  Time: " + hour_of_the_day+":"+minute_of_the_day)
            fig.savefig(dir_name + str(i) + '.png', dpi=150, format='png', bbox_inches='tight')
            plt.close()

def viz_pred_incident(speed, flow, occ, incident, low_qual_idx, pred_vec, event_num, month_num, dir_name):
    """
    visualize all the incidents in the incident Dataframe

    speed: speed Dataframe (either normalized or original)
    flow: volume Dataframe (either normalized or original)
    occ: occupancy Dataframe (either normalized or original)
    incident: incident Dataframe.
    """
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', "Sun"]
    sensors = speed.columns
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i in range(incident.shape[0]):
        if i in low_qual_idx:
            print("Took out {} example for it's low quality".format(i))
            pass
        else:
            vis_idx = np.where(event_num == (0.5*(i + month_num + 1) * (i + month_num) + month_num))[0]
            pred_temp = pred_vec[vis_idx]
            detected_area = np.where(pred_temp == 1)[0]
            

            date = pd.to_datetime(incident.iloc[i]['Time']) - pd.Timedelta(3600, unit='s') 
            times = pd.to_datetime(incident.iloc[i]['Time'])
            date2 = times + pd.Timedelta(2*3600, unit='s')

            time_span = speed.loc[date:date2].index

            num = i    
            duration = incident.iloc[num]['Duration']*60
            upstr_node_pos = incident.iloc[num]['Upstream ID']
            downstr_node_pos = incident.iloc[num]['Downstream ID']
            fig, ax = plt.subplots(1, 3, figsize=(21, 3))
            ax[0].plot(speed.loc[date:date2, upstr_node_pos], label='upstream')
            ax[0].plot(speed.loc[date:date2, downstr_node_pos], label='downstream')
            if duration <= 2*3600:
                incident_end_time = times + pd.Timedelta(duration, unit='s')
            else:
                incident_end_time = date2
            try:
                ax[0].axvspan(times, incident_end_time, color='red', alpha=0.1)
                for j in range(len(detected_area)):
                    ax[0].axvline(time_span[detected_area[j]], color='green', alpha=0.1)

            except:
                pass
            ax[0].set_ylabel('speed')
            ax[0].grid('off')
            ax[0].legend()

            ax[1].plot(flow.loc[date:date2, upstr_node_pos], label='upstream')
            ax[1].plot(flow.loc[date:date2, downstr_node_pos], label='downstream')
            try:
                ax[1].axvspan(times, incident_end_time,  color='red', alpha=0.1) #times + pd.Timedelta(duration, unit='s'),
            except:
                pass
            ax[1].set_ylabel('flow')
            ax[1].grid('off')
            ax[1].legend()

            ax[2].plot(occ.loc[date:date2, upstr_node_pos], label='upstream')
            ax[2].plot(occ.loc[date:date2, downstr_node_pos], label='downstream')
            try:
                ax[2].axvspan(times, incident_end_time, color='red', alpha=0.1)
            except:
                pass
            ax[2].grid('off')
            ax[2].set_ylabel('occupancy')
            ax[2].legend()
            day_of_the_week = days[int(pd.Timestamp(times).dayofweek)]
            hour_of_the_day = str(pd.Timestamp(times).hour)
            minute_of_the_day = str(pd.Timestamp(times).minute)
            fig.suptitle("Weekday: " + day_of_the_week + "  Time: " + hour_of_the_day+":"+minute_of_the_day)
            fig.savefig(dir_name + str(i) + '.png', dpi=150, format='png', bbox_inches='tight')
            plt.close()


class MinMax:
    """
    MinMax scalar
    """
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
    def transform(self, x):
        x = (x - self.minimum)/(self.maximum - self.minimum) + self.minimum
        return x

def group_slices(sample, event_num_vec, span, avg_steps): 
    """
    to group the predicted time slices to form inidividual incidents.

    sample: predicted vector
    span: the span of each continuous time period in number of timestamps
    avg_steps: the average duration of incidents in the training set or maximum duration of grouped incident

    return the grouped predicted vector
    """ 
    unique_dates = np.unique(event_num_vec)
    for date in unique_dates:
        idx_event = np.where(event_num_vec == date)[0]
        temp = sample[idx_event]
        for i in range(len(temp)):
            if temp[i] == 1 and temp[i:i + 120].any() == 1:
                idx = np.where(temp[i:i + 120]==1)[0][-1]
                sample[idx_event[0] + i: idx_event[0] + i + idx] = 1
    groups, num_groups = label(sample)
    idx = [np.where(groups == i+1)[0] for i in range(num_groups)]
    for grp in idx:
        if len(grp) > avg_steps:
            #print('exceeding norm')
            sample[grp[avg_steps:]] = 0


    # for j in range(len(sample)//span): 
    #     for i in range(len(sample[j*span:(j+1)*span])):
    #         temp = sample[j*span:(j+1)*span]
    #         if temp[i] == 1 and temp[i:i+120].any() == 1: 
    #             idx = np.where(temp[i:i+120]==1)[0][-1]                          
    #             sample[j*span+i:j*span+i+idx] = 1 
    # groups, num_groups = label(sample)
    # idx = [np.where(groups == i+1)[0] for i in range(num_groups)]
    # for grp in idx:
    #     if len(grp) > avg_steps:
    #         sample[grp[avg_steps:]] = 0
    
    return sample 

def cal_metrics(pred, true):
    """
    calculate metrics

    pred: grouped predicted vector
    true: ground truth vector

    return: number of correctly detected incident, number of true incidents, number of predicted incident, and mean time to detect
    """
    true_label, true_num = label(true)
    pred_label, pred_num = label(pred)
    true_idx = [np.where(true_label == i+1)[0] for i in range(true_num)] 
    pred_idx = [np.where(pred_label == i+1)[0] for i in range(pred_num)]
    correct_detected = 0
    time_to_detect = 0
    pred_dect_num = 0
    for v in true_idx:
        indicator = 0 # for each real incident reset this to indicate if there is correct detection
        for i, u in enumerate(pred_idx):
            if u.min() <= v.max() + 20  and u.max() >= v.min() - 20   : # determine soft overlapping
                indicator += 1
                if indicator == 1: # only take the first correct one to calculate time
                    time_to_detect += u.min() - v.min()
            else:
                pred_dect_num += 1
        if indicator > 0:
            correct_detected += 1
            pred_dect_num += 1


    return correct_detected, true_num, pred_dect_num, time_to_detect*30/60





def scores(y_pred_fil, y_true, event_num_vec):
    """
    print out the scores calculated in the cal_matrics function
    """
    num_incidents = 0 
    correct = 0
    pred_incidents = 0
    time = []
    unique_dates, order = np.unique(event_num_vec, return_index=True)
    order = np.argsort(order)
    unique_dates = unique_dates[order]
    for date in unique_dates:
        event_idx = np.where(event_num_vec == date)[0]
        correct_tmp, true_num_tmp, pred_num_tmp, t = cal_metrics(y_pred_fil[event_idx], y_true[event_idx])
        correct += correct_tmp
        num_incidents += true_num_tmp
        pred_incidents += pred_num_tmp
        time.append(t)
    print('Detection Rate is', correct/num_incidents)
    print('False Alarm Rate is', (pred_incidents - correct)/pred_incidents)
    print('Mean Time to Detect is', np.mean(time))
    print('Actual number of incidents', num_incidents)
    print('Detected number of incidents', pred_incidents)
    print('Correctly detected incidents', correct)
    
class real_time_peak_detection():
    def __init__(self, array, lag, threshold, influence):
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag:
            return 0
        
        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]

        if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = 0

            self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return self.signals[i]
    
def filter_detection(signals, free_idx):
    marking = np.array(signals).copy()
    marking[free_idx] = 0
    groups, num_groups = label(marking)
    idx = [np.where(groups == k+1)[0] for k in range(num_groups)]
    for grp in idx:
        if len(grp) <= 20:
            marking[grp] = 0
    return marking

# def cal_metrics(pred, true):
#     """
#     calculate metrics

#     pred: grouped predicted vector
#     true: ground truth vector

#     return: number of correctly detected incident, number of true incidents, number of predicted incident, and mean time to detect
#     """
#     true_label, true_num = label(true)
#     pred_label, pred_num = label(pred)
#     true_idx = [np.where(true_label == i+1)[0] for i in range(true_num)] 
#     pred_idx = [np.where(pred_label == i+1)[0] for i in range(pred_num)]
#     correct_detected = 0
#     time_to_detect = 0
#     for i, u in enumerate(pred_idx):
#         for v in true_idx:
#             if u.min() <= v.max() + 20 and u.max() >= v.min() - 20: # determine soft overlapping
#                 correct_detected += 1
#                 time_to_detect += u.min() - v.min()


#     return correct_detected, true_num, pred_num, time_to_detect


# def scores(y_pred_fil, y_true, event_num_vec):
#     """
#     print out the scores calculated in the cal_matrics function
#     """
#     num_incidents = 0 
#     correct = 0
#     pred_incidents = 0
#     time = []
#     unique_dates = np.unique(event_num_vec)
#     for date in unique_dates:
#         event_idx = np.where(event_num_vec == date)[0]
#         correct_tmp, true_num_tmp, pred_num_tmp, t = cal_metrics(y_pred_fil[event_idx], y_true[event_idx])
#         correct += correct_tmp
#         num_incidents += true_num_tmp
#         pred_incidents += pred_num_tmp
#         time.append(t)
#     print('Detection Rate is', correct/num_incidents)
#     print('False Alarm Rate is', (pred_incidents - correct)/pred_incidents)
#     print('Mean Time to Detect is', np.mean(time))
#     print('Actual number of incidents', num_incidents)
#     print('Detected number of incidents', pred_incidents)
#     print('Correctly detected incidents', correct)

def viz_prediction(pred, true, span, dir_):
    """
    visualize the detected incident against the true incidents
    """
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    for i in range(len(true)//span):
        section = np.arange(span)
        plt.figure(figsize=(20,5))
        plt.plot(true[span*i:span*(i+1)], linewidth=3)
        plt.plot(pred[span*i:span*(i+1)], alpha=0.7)
        print(i, np.unique(true[360*i:360*(i+1)], return_counts=True))
        plt.savefig(dir_ + str(i) + '.png', dpi=150, format='png', bbox_inches='tight')
        plt.close()

def check_quality(incident, speed, flow, occ):
    """
    check data quality
    
    incident: incident dataframe with up and down stream info
    speed: normalized speed df
    flow: normalized flow df
    occ: normalized occupancy df

    Returns indices of sample instances with low data quality
    """
    low_qual_idx = []
    for i in range(incident.shape[0]):
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
        if np.mean(ddt_speed) <= (0.01566702 - 0.01043915) or np.mean(ddt_flow) <= (0.03229781 - 0.02578795) or np.mean(ddt_occ) <= (0.04392279 - 0.04089887):
            low_qual_idx.append(i)
            print("incident with ID {} might be low quality data".format(i))
    return low_qual_idx



def get_detection(rnn_output_prob, x, event_num, diff_threshold, methods="derivative"):
    unique_dates, order = np.unique(event_num, return_index=True)
    order = np.argsort(order)
    unique_dates = unique_dates[order]
    detection_vector = []
    diff_vec = []
    prob_vec = []
    speed_up_vec = []
    speed_down_vec = []
    for i, date in enumerate(unique_dates):
        idx_event = np.where(event_num == date)[0]
        #current_profile = profiles[i]

        speed_up = x[idx_event, 0, -1]
        speed_down = x[idx_event, 3, -1]
        
        speed_up_vec.append(speed_up)
        speed_down_vec.append(speed_down)

        up_idx = np.where(speed_up > 0.8)[0]
        down_idx = np.where(speed_down > 0.8)[0]
        free_idx = np.intersect1d(up_idx, down_idx)
     
        prob_vec.append(rnn_output_prob[idx_event])
        current_prob_out = rnn_output_prob[idx_event]
        prob_mask = np.where(current_prob_out > 0.07)[0] # upper threshold for incidents
        prob_mask_false = np.where(current_prob_out < 0.03)[0] # threshold for nonincidents
        if methods == "derivative":
            diff = np.diff(rnn_output_prob[idx_event])
            diff = np.insert(diff, 0, np.nan)
            diff_vec.append(diff)

            mask = rnn_output_prob[idx_event] >= 0.5
            diff[~mask] = np.nan

            detection = np.where(diff >= diff_threshold)[0]

            detected_start = np.zeros(diff.shape)
            detected_start[detection] = 1
            detected_start[prob_mask] = 1
            detected_start[prob_mask_false] = 0
            detected_start[free_idx] = 0

            detection_vector.append(detected_start)
        else:
            detector = real_time_peak_detection(current_prob_out[:30], 5, 8, 0) # there are parameters to tune
            for j, d in enumerate(current_prob_out[30:]):
                a = detector.thresholding_algo(d)
        
            detected_start = filter_detection(detector.signals, free_idx)
            detection_vector.append(detected_start.reshape(-1, 1))
        
    return detection_vector

