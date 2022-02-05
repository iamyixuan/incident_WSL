import numpy as np
import h5py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from utils import filter_detection, scores, cal_metrics
from utils import MinMax
from sklearn.metrics import classification_report, confusion_matrix


with h5py.File('../data/lstm_norm_data/long_horizon/final/train/snorkel_train_5.h5','r') as f:
    x_train = f['input'][...]
    y_train = f['output'][...]
    #train_dist = f['sensor_dist'][...]
    #event_num_train = f['event)num'][...]


y_train = np.where(y_train > 0.5, 1, 0)

print(np.unique(y_train, return_counts=True))
with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/val/val_for_snorkel.h5','r') as f:
    x_val = f['input'][...]
    y_val = f['output'][...]

horizon = 10
steps = int(horizon * 60 / 30)

def add_diff(x):
    speed_diff = (x[:, 3, :] - x[:, 0, :]) / 0.5  * (x[:, 3, :] + x[:, 0, :])
    occ_diff = (x[:, 5, :] - x[:, 2, :]) / 0.5 * (x[:, 5, :] + x[:, 2, :])
    speed_diff = np.expand_dims(speed_diff, axis=1)
    occ_diff = np.expand_dims(occ_diff, axis=1)
    x = np.concatenate([x, speed_diff, occ_diff], axis=1)
    return x

x_train = add_diff(x_train)
x_val = add_diff(x_val)

x_train = np.concatenate([x_train])
y_train = np.concatenate([y_train])

print(np.unique(y_train, return_counts=True))
print(x_train.shape)

x_val = np.concatenate([x_val])
y_val = np.concatenate([y_val])

profiles_use = [0, 2, 3, 5]
x_train = np.transpose(x_train, axes=[0, 2, 1])[:, -steps:, profiles_use]
x_val = np.transpose(x_val, axes=[0, 2, 1])[:, -steps:, profiles_use]

sclr = MinMax(np.min(x_train, axis=(0, 1)), np.max(x_train, axis=(0, 1)))
x_train = sclr.transform(x_train)
x_val = sclr.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)

rf  = RandomForestClassifier()
rf.fit(x_train, y_train)



val_pred = rf.predict(x_val)



print("validation report is \n", classification_report(y_val, val_pred))

# print("acc", a, b)
# print("f1", f1_score(y_train, c), f1_score(y_val, d))

# print("pred", d[:5])
# unique_dates, order = np.unique(event_num_val, return_index=True)
# order = np.argsort(order)
# unique_dates = unique_dates[order]

# detection_array = []
# speed_up_vec = []
# speed_down_vec = []
# for i, date in enumerate(unique_dates):
#     idx_event = np.where(event_num_val == date)[0]
#     speed_up = x_val[idx_event, 0, -1]
#     speed_down = x_val[idx_event, 3, -1]

#     speed_up_vec.append(speed_up)
#     speed_down_vec.append(speed_down)

#     up_idx = np.where(speed_up > 0.8)[0]
#     down_idx = np.where(speed_down > 0.8)[0]
#     free_idx = np.intersect1d(up_idx, down_idx)
#     detections = d[idx_event]
#     detections = filter_detection(detections, free_idx)
#     detection_array.append(detections.reshape(-1 ,1))


# final_detection = np.concatenate(detection_array).reshape(-1, 1)
# print(scores(final_detection, y_val, event_num_val))