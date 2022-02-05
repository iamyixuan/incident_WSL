import os
import glob
import numpy as np
import pandas as pd

def match_incidents(sensor_file_name, incident_file_name):
    """
    To extract incident information and match it with the corresponding
    detecting sensors and timestamps

    sensor_file_name: xlsx file downloaded from PeMS
    incident_file_name: Dataframe in h5 containing the incident information of desired period.

    Return a Dataframe with incident starting time. duration, and correpsonding upstream & downstream  sensor IDs
    """
    sensors_file = pd.read_excel(sensor_file_name)
    #sensors_file = sensors_file[sensors_file['District'] == 4] # use this line if want to specify district
    accidents = pd.read_hdf(incident_file_name)
    extract_incident = []
    sensor_pm = sensors_file['Abs PM'].values
    sensor_id = sensors_file['ID'].values
    for i in range(accidents.shape[0]):
        accident_pm = accidents.iloc[i]['Abs PM']
        if accident_pm >= sensor_pm[0] and accident_pm <= sensor_pm[-1]:
            closest = np.argsort(np.abs(accident_pm - sensor_pm))[0]
            if sensor_pm[closest] > accident_pm:
                up = sensor_id[closest - 1]
                down = sensor_id[closest]
            else:
                up = sensor_id[closest]
                down = sensor_id[closest + 1]
            extract_incident.append([accidents['Start Time'].iloc[i], accidents['Duration (mins)'].iloc[i], up, down])

    incidents = pd.DataFrame(extract_incident, columns=['Time', 'Duration', 'Upstream ID', 'Downstream ID'])
    return incidents



def aggregate_raw(raw_file_path, target_dir, sensor_list):
    """
    To process and aggregate lane-wise raw sensor measurement files.
    It creates 3 h5 files (speed, volum, occupancy) per day.

    raw_file_path: directory containing the raw files which is by day.
    target_dir: target directory where the processed files can be stored.
    sensor_list: a list of sensor IDs of which the information is wanted.
    """

    fnames = glob.glob(raw_file_path + '*.txt')
    dist_info = fnames[0][:-9]
    # read the text file with sensor ids
    for idx, filename in enumerate(fnames):
        print(filename)
        day = int(filename[-6:-4])
        month = filename[-9:-7]
        day_df = pd.read_csv(filename, index_col=None, header=None)
        flow_seq = np.array([2, 5, 8, 11, 14, 17, 20, 23])
        occ_seq = flow_seq + 1
        speed_seq = occ_seq + 1
        flow_seq = np.append([1], flow_seq)
        occ_seq = np.append([1], occ_seq)
        speed_seq = np.append([1], speed_seq)
        if day < 9:
            start= month + '/0' + str(day) + '/2019'
            end = start +' 23:59:30'
        elif day == 9: 
            start= month+ '/0' + str(day) + '/2019'
            end = start +' 23:59:30'
        else:
            start= month+ '/' + str(day) + '/2019'
            end = start +' 23:59:30'
            
        date_index = pd.date_range(start, end, freq='30s')
        df_flow = pd.DataFrame(index=date_index)
        df_occ = pd.DataFrame(index=date_index)
        df_speed = pd.DataFrame(index=date_index)
        for sn_num in range(len(sensor_list)):
            df = day_df[day_df[1] == sensor_list[sn_num]]
            df.index = pd.to_datetime(df[0])
            if df.shape[0] > 0 :
                del df[0]
                delta = df.index[0] - date_index[0]
                df.index = df.index - delta
                date_index = pd.date_range(start, end, freq='30s')
                df_new = pd.DataFrame(index=date_index)
                df_new = df_new.join(df)
                df_new = df_new.interpolate(method='linear', limit_direction='forward', axis=0)
                df_new = df_new.ffill().bfill()
                lane_number = sum(~np.isnan(df_new[flow_seq].mean().values)) - 1
                df_flow_temp = df_new[flow_seq].iloc[:, :lane_number+1]
                df_flow[sensor_list[sn_num]] = df_flow_temp.iloc[:, 1:].sum(axis=1)
                df_occ_temp = df_new[occ_seq].iloc[:, :lane_number+1]
                df_occ[sensor_list[sn_num]] = df_occ_temp.iloc[:, 1:].sum(axis=1)/lane_number
                df_sp_temp = df_new[speed_seq].iloc[:, :lane_number+1]
                df_flow_temp.columns = list(range(1,lane_number+2))
                df_sp_temp.columns = list(range(1,lane_number+2))
                df_mult = df_sp_temp.iloc[:, 1:] * df_flow_temp.iloc[:, 1:]
                df_sum = df_flow_temp.iloc[:, 1:].sum(axis=1)
                df_speed[sensor_list[sn_num]] = df_mult.div(df_sum, axis=0).sum(axis=1)
                df_flow.to_hdf(target_dir + fnames[0][-35:-31] + 'flow_2019_' + month + '_0' + str(day) +'.h5', key='df') 
                df_occ.to_hdf(target_dir + fnames[0][-35:-31] + 'occ_2019_' + month + '_0' + str(day) +'.h5', key='df') 
                df_speed.to_hdf(target_dir + fnames[0][-35:-31] + 'speed_2019_' + month + '_0' + str(day) +'.h5', key='df')
            else:
                print("No matching sensor information in the raw data!") 
                print("Sensor num", sn_num)
            

def concat_info_h5(file_dir, target_dir, name):
    """
    concatenate days of sensor measurements together
    
    file_dir: directory containing single-type aggregated sensor measurements.
    target_dir: target directory to save the processed files
    name: file name of the processed file.
    """
    try:
        os.remove(file_dir + '/.DS_store')
    except:
        pass
    
    filenames = os.listdir(file_dir)
    for i, df_name in enumerate(filenames):
        if i == 0:
            df = pd.read_hdf(file_dir + df_name)
        else:
            df = pd.concat([df, pd.read_hdf(file_dir + df_name)])
    df_sorted = df.sort_index()
    df_sorted.to_hdf(target_dir + name +'.h5', key='df')

def concat_info_xlxs(file_dir, target_dir, name):
    """
    concatenate days of sensor measurements together
    
    file_dir: directory containing single-type aggregated sensor measurements.
    target_dir: target directory to save the processed files
    name: file name of the processed file.
    """
    filenames = glob.glob(file_dir + '*.xlsx')
    for i, df_name in enumerate(filenames):
        if i == 0:
            df = pd.read_excel(df_name)
        else:
            df = pd.concat([df, pd.read_excel(df_name)])
    df_sorted = df.sort_index()
    df_sorted.to_hdf(target_dir + name +'.h5', key='df')

def fill_smooth_norm(speed, flow, occ):
    """
    clean, smooth, and normalize the time series data

    speed_file: processed speed file name, in h5.
    volume_file: processed volume/flow file name, in h5.
    occ_file: processed occupancy file name, in h5.
    """

    # forward fill 0 values (no measurements)
    mask_speed = speed.values==0
    mask_flow = flow.values==0
    mask_occ = occ.values==0

    speed.iloc[mask_speed] = np.nan
    flow.iloc[mask_flow] = np.nan
    occ.iloc[mask_occ] = np.nan
    speed = speed.fillna(method='ffill', axis=0) # only forward fill speed
    flow = flow.fillna(method='ffill', axis=0)
    occ = occ.fillna(method='ffill', axis=0)
    speed = speed.interpolate(axis=1)
    flow = flow.interpolate(axis=1)
    occ = occ.interpolate(axis=1)

    """
    smoothening & normalization
    """

    speed = speed.ewm(span=5).mean()
    flow = flow.ewm(span=5).mean()
    occ = occ.ewm(span=5).mean()

    free_speed = speed.quantile(0.95)
    free_flow = flow.quantile(0.95)
    free_occ = occ.quantile(0.95)

    norm_speed = speed/free_speed
    norm_flow = flow/free_flow
    norm_occ = occ/free_occ
    return norm_speed, norm_flow, norm_occ, speed, flow, occ

def filter_on_duration(df):
    """
    fitler out incidents of which the duration is less than 10 min
    
    df: incident Dataframe
    """
    tmp = df.copy()
    tmp = tmp[tmp['Duration']>=10]
    return tmp

def filter_by_speed(inci_df, speed_df):
    """
    filter based on speed diff
    5 m/h
    
    filter out the flat speed profiles as well
    
    if at the incident marked starting time
    the upstream speed is not going down
    treat it as a human error (5 min slack)
    """
    idx_to_save = []
    for i in range(inci_df.shape[0]):
        up_sensor = inci_df.iloc[i]['Upstream ID']
        down_sensor = inci_df.iloc[i]['Downstream ID']
        inci_start = pd.to_datetime(inci_df.iloc[i]['Time'])
        five_min_time = inci_start + pd.Timedelta(5*60, unit='s')
        #sp_10min = speed.loc[inci_start - pd.Timedelta(10*60, unit='s'):inci_start, up_sensor].mean()  
        if (up_sensor in speed_df.columns) and (down_sensor in speed_df.columns): 
            sp_5min_aftr_up = speed_df.loc[inci_start:five_min_time, up_sensor]
            sp_5min_aftr_down = speed_df.loc[inci_start:five_min_time, down_sensor]
            diff = sp_5min_aftr_down.mean() - sp_5min_aftr_up.mean()
            if len(np.unique(sp_5min_aftr_up)) == 1 or len(np.unique(sp_5min_aftr_down)) == 1:
                print("Flat speed profile!")
                pass# take out the flat speed profiles
            elif diff > 5:
                idx_to_save.append(i)
            else:
                print("Incident not satisfying the condition!")
        else:
            pass
    return inci_df.iloc[idx_to_save]

def filter_by_free_flow_speed(inci_df, speed_df):
    """
    need to fill out the note
    """
    idx_to_save = []
    for i in range(inci_df.shape[0]):
        up_sensor = inci_df.iloc[i]['Upstream ID']
        down_sensor = inci_df.iloc[i]['Downstream ID']
        duration = inci_df.iloc[i]['Duration']
        inci_start = pd.to_datetime(inci_df.iloc[i]['Time'])
        interval = inci_start + pd.Timedelta(duration*60, unit='s')

        #sp_10min = speed.loc[inci_start - pd.Timedelta(10*60, unit='s'):inci_start, up_sensor].mean()  
        if (up_sensor in speed_df.columns) and (down_sensor in speed_df.columns): 
            up_stream_speed = speed_df.loc[inci_start : interval, up_sensor]
            down_stream_speed = speed_df.loc[inci_start : interval, down_sensor]
            if_free_up = np.all(up_stream_speed >=0.8)
            if_free_down = np.all(down_stream_speed >= 0.8)
            if len(np.unique(up_stream_speed)) == 1 or len(np.unique(down_stream_speed)) == 1:
                print("Flat speed profile!")
                pass# take out the flat speed profiles
            elif if_free_down and if_free_up:
                print('Free Flow')
                pass
            else:
                idx_to_save.append(i)
        else:
            pass
    return inci_df.iloc[idx_to_save]