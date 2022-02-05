import numpy as np
import pandas as pd

"""
To correct the human error manually
"""

incident_df = pd.read_hdf('../data/2019_accidents/2019_02_matched.h5')

to_use = incident_df.copy()
print(to_use.shape)

#to_use['Time'].iloc[12] = str(pd.to_datetime(incident_df['Time'].iloc[12]) - pd.Timedelta(15, unit='m'))
# to_use['Time'].iloc[19] = str(pd.Timestamp(incident_df['Time'].iloc[19]) - pd.Timedelta(30, unit='m'))
# to_use['Time'].iloc[20] = str(pd.Timestamp(incident_df['Time'].iloc[20]) - pd.Timedelta(10, unit='m'))
# to_use['Time'].iloc[22] = str(pd.Timestamp(incident_df['Time'].iloc[22]) - pd.Timedelta(15, unit='m'))
# to_use['Time'].iloc[30] = str(pd.Timestamp(incident_df['Time'].iloc[30]) - pd.Timedelta(40, unit='m'))
# to_use['Time'].iloc[31] = str(pd.Timestamp(incident_df['Time'].iloc[31]) - pd.Timedelta(45, unit='m'))
# to_use['Time'].iloc[35] = str(pd.Timestamp(incident_df['Time'].iloc[35]) - pd.Timedelta(45, unit='m'))

to_use.drop(index=to_use.iloc[36].name, inplace=True)
to_use.to_hdf('../data/2019_accidents/2019_02_corrected.h5', key='df')
print(to_use.shape)