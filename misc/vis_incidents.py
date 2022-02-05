import numpy as np
import os
import h5py
import pandas as pd
from utils import viz_incidents, viz_pred_incident
import matplotlib as mpl
import matplotlib.pyplot as plt

from preprocesing import fill_smooth_norm, filter_on_duration, filter_by_speed
from utils import check_quality
mpl.rcParams["font.size"] = 14
#mpl.rcParams["font.weight"] = "bold"
fontsize = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.edgecolor'] = 'black'



"""
Load speed flow occupancy dataframes (normalized)
Load incident dataframe (fitlered)
"""
speed = pd.read_hdf('../data/processed_2019/Nov_concat/2019_11_speed.h5')
flow = pd.read_hdf('../data/processed_2019/Nov_concat/2019_11_flow.h5')
occ = pd.read_hdf('../data/processed_2019/Nov_concat/2019_11_occupancy.h5')

incident_df = pd.read_hdf('../data/2019_accidents/2019_11_matched.h5')
#incident_df = pd.read_csv('../data/processed_2020_jan_feb_march/corrected_incident_test.csv')

norm_speed, norm_flow, norm_occ, speed, flow, occ = fill_smooth_norm(speed, flow, occ)

"""
correcting human errors
and drop the garbage profiles
"""
incident_filterd = filter_on_duration(filter_by_speed(incident_df, speed))
low_qual_idx = check_quality(incident_filterd, norm_speed, norm_flow, norm_occ)


print('The number of remaining incidents is', incident_filterd.shape[0])
inci_cor = incident_filterd.copy()

# inci_cor['Time'].iloc[4] = pd.to_datetime(incident_filterd.iloc[4]['Time']) - pd.Timedelta(30*60, unit='s') 
# inci_cor['Time'].iloc[9] = pd.to_datetime(incident_filterd.iloc[9]['Time']) - pd.Timedelta(40, 'm') 
# inci_cor['Time'].iloc[30] = pd.to_datetime(incident_filterd.iloc[30]['Time']) + pd.Timedelta(50, 'm') 

# inci_cor.to_hdf('../data/2019_accidents/2019_05_corrected.h5', key='df')
# filter the incident based on speed and duration criteria.

#viz_incidents(norm_speed, norm_flow, norm_occ, inci_cor, low_qual_idx,  '../notes/plots/2019_06_incidents/')

print(incident_filterd)




def viz_pred_incident_new(speed, flow, occ, incident, low_qual_idx, pred_vec, event_num, month_num, dir_name):
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
            #axis_len = len(speed.loc[date:date2, upstr_node_pos])

            fig, ax = plt.subplots(1, 3, figsize=(21, 3))
            ax[0].plot(speed.loc[date:date2, upstr_node_pos], linewidth=3, label='Upstream')
            ax[0].plot(speed.loc[date:date2, downstr_node_pos], linewidth=3, label='Downstream')
            ax[0].set_xticklabels([])
            ax[0].set_xlabel('Time (30s)')
            if duration <= 2*3600:
                incident_end_time = times + pd.Timedelta(duration, unit='s')
            else:
                incident_end_time = date2
            try:
                ax[0].axvspan(times, incident_end_time, color='red', alpha=0.1)
                # for j in range(len(detected_area)):
                #     ax[0].axvline(time_span[detected_area[j]], color='green', alpha=0.1)

            except:
                pass
            ax[0].set_ylabel('Normalized Speed')
            ax[0].grid(False)
            ax[0].legend()

            ax[1].plot(flow.loc[date:date2, upstr_node_pos], linewidth=3, label='Upstream')
            ax[1].plot(flow.loc[date:date2, downstr_node_pos], linewidth=3, label='Downstream')
            ax[1].set_xticklabels([])
            ax[1].set_xlabel('Time (30s)')
            try:
                ax[1].axvspan(times, incident_end_time,  color='red', alpha=0.1) #times + pd.Timedelta(duration, unit='s'),
                # for j in range(len(detected_area)):
                #     ax[1].axvline(time_span[detected_area[j]], color='green', alpha=0.1)
            except:
                pass
            ax[1].set_ylabel('Normalized Volume')
            ax[1].grid(False)
            ax[1].legend()

            ax[2].plot(occ.loc[date:date2, upstr_node_pos], linewidth=3, label='Upstream')
            ax[2].plot(occ.loc[date:date2, downstr_node_pos], linewidth=3, label='Downstream')
            ax[2].set_xticklabels([])
            ax[2].set_xlabel('Time (30s)')
            try:
                ax[2].axvspan(times, incident_end_time, color='red', alpha=0.1)
                # for j in range(len(detected_area)):
                #     ax[2].axvline(time_span[detected_area[j]], color='green', alpha=0.1)
            except:
                pass
            ax[2].grid(False)
            ax[2].set_ylabel('Normalized Occupancy')
            ax[2].legend()
            day_of_the_week = days[int(pd.Timestamp(times).dayofweek)]
            hour_of_the_day = str(pd.Timestamp(times).hour)
            minute_of_the_day = str(pd.Timestamp(times).minute)
            # fig.suptitle("Weekday: " + day_of_the_week + "  Time: " + hour_of_the_day+":"+minute_of_the_day)
            fig.savefig(dir_name + str(i) + '.png', dpi=150, format='png', bbox_inches='tight')
            plt.close()


"""
visualize with prediction
"""

with h5py.File('../data/lstm_norm_data/test/lstm_norm_convolve_2019_11.h5', 'r') as f:
    event_num = f['event_num'][...]
pred = np.loadtxt('../data/saved_predictions/pred_2019_11.txt')

viz_pred_incident_new(norm_speed, norm_flow, norm_occ, incident_filterd, low_qual_idx, pred, event_num, 19+12, '../notes/plots/2019_11_paper_plots/')