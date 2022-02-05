import numpy as np
import h5py
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from utils import MinMax, scores, group_slices

# with h5py.File("../data/lstm_norm_data/long_horizon/final/train/train.h5", "r") as f:
#         x_train_1 = f["x_train"][:, (2, 5), -1]
#         y_train_1 = f["y_train"][...]
#         x_val_1 = f["x_val"][:, (2, 5), -1]
#         y_val_1 = f["y_val"][...]

# with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/train/pickedTrainDoubleDrop.h5','r') as f:
#     x_train = f['input'][:, (2, 5), -1]
#     y_train = f['output'][...]
#     train_dist = f['sensor_dist'][...]
#     #event_num_train = f['event)num'][...]

with h5py.File('../data/lstm_norm_data/long_horizon/final/train/snorkel_train_5.h5','r') as f:
    x_train = f['input'][:, (2, 5), -1]
    y_train = f['output'][...]
    #train_dist = f['sensor_dist'][...]
    #event_num_train = f['event)num'][...]
y_train = np.where(y_train > 0.5, 1, 0)
with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/val/val_for_snorkel.h5','r') as f:
    x_val = f['input'][:, (2, 5), -1]
    y_val = f['output'][...]
    val_dist = f['sensor_dist'][...]
    #event_num_val = f['event)num'][...]

x_train = np.concatenate([x_train])
y_train = np.concatenate([y_train])

x_val = np.concatenate([x_val])
y_val = np.concatenate([y_val])


def get_thresholds(occ, labels):
    incident_mask = np.where(labels == 1)[0]
    occ_use = occ[incident_mask]
    T1 = np.mean(occ_use[:, 0] - occ_use[:, 1])
    T2 = np.mean((occ_use[:, 0] - occ_use[:, 1])/(occ_use[:, 0] + 1e-5))
    T3 = np.mean((occ_use[:, 0] - occ_use[:, 1])/(occ_use[:, 1] + 1e-5))
    return T1, T2, T3


T1, T2, T3 = get_thresholds(x_train, y_train)
print(T1, T2, T3)

def california_detect(test_occ, T1, T2, T3):
    occ_diff = test_occ[:, 0] - test_occ[:, 1]
    occ_ratio1 = (test_occ[:, 0] - test_occ[:, 1])/(test_occ[:, 0] + 1e-5)
    occ_ratio2 = (test_occ[:, 0] - test_occ[:, 1])/(test_occ[:, 1] + 1e-5)

    pred = np.zeros(test_occ.shape[0])

    pred = np.where((occ_diff >= T1) & (occ_ratio1 >= T2) & (occ_ratio2 >= T3), 1, 0)
    return pred



y_pred = california_detect(x_val, T1, T2, T3)
#test_pred = california_detect(x_test, T1, T2, T3)

print('val report is \n ', classification_report(y_val, y_pred))
# print('test report is \n ', classification_report(y_test, test_pred))
# print('test confusion matrix is \n ', confusion_matrix(y_test, test_pred))

# num_detected = np.sum(y_pred)
# num_incident = np.sum(labels_test)

# num_correct = len(np.where((y_pred + labels_test) == 2)[0])

# print(num_detected)

# print('Detection Rate', num_correct/num_incident)
# print('False Alarm Rate', (num_detected - num_correct)/num_detected, '\n')



# y_pred_fil = group_slices(y_pred, event_num, 360, 160)

# scores(y_pred_fil, labels_test, event_num)

