import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from numpy.core.defchararray import add

from utils import MinMax, scores, group_slices
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report, confusion_matrix

# with h5py.File("../data/lstm_norm_data/long_horizon/hand_pick.h5") as handpick:
#     x_train = handpick["x_train"][...]
#     y_train = handpick["y_train"][...]
#     x_val = handpick["x_val"][...]
#     y_val = handpick["y_val"][...]

# with h5py.File('../data/lstm_norm_data/long_horizon/final/test/test.h5', 'r') as f:
#     x_test = f['input'][...]
#     y_test = f['output'][...]
#     event_num_test = f['event_num'][...]

# def add_diff(x):
#     speed_diff = (x[:, 3, :] - x[:, 0, :]) / 0.5 * (x[:, 3, :] + x[:, 0, :])
#     occ_diff = (x[:, 5, :] - x[:, 2, :]) / 0.5 * (x[:, 5, :] + x[:, 2, :])
#     speed_diff = np.expand_dims(speed_diff, axis=1)
#     occ_diff = np.expand_dims(occ_diff, axis=1)
#     x = np.concatenate([x, speed_diff, occ_diff], axis=1)
#     return x


# horizon = 10

# steps = int(horizon * 60 / 30)

# x_test = add_diff(x_test)


# profile_to_use = [0, 3, 7]
# x_train = np.transpose(x_train, axes=[0, 2, 1])[:, -steps:, profile_to_use]
# x_val = np.transpose(x_val, axes=[0, 2, 1])[:, -steps:, profile_to_use]
# x_test = np.transpose(x_test, axes=[0, 2, 1])[:, -steps:, profile_to_use]


# x_train = x_train.reshape(x_train.shape[0], -1)
# x_val = x_val.reshape(x_val.shape[0], -1)
# x_test = x_test.reshape(x_test.shape[0], -1)


# sclr = MinMax(np.min(x_train, axis=(0, 1)), np.max(x_train, axis=(0, 1)))
# x_train = sclr.transform(x_train)
# x_val = sclr.transform(x_val)
# x_test = sclr.transform(x_test)


# y_train = y_train
# print(np.unique(y_train, return_counts=True))
# y_val = y_val
# y_test = y_test



# shuffle_idx = np.random.permutation(len(x_train))
# x_train = x_train[shuffle_idx]
# y_train = y_train[shuffle_idx]


# with h5py.File("../data/lstm_norm_data/long_horizon/hand_pick.h5", "r") as f:
#         x_train_1 = f["x_train"][...]
#         y_train_1 = f["y_train"][...]
#         x_val_1 = f["x_val"][...]
#         y_val_1 = f["y_val"][...]

# with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/train/pickedTrainDoubleDrop.h5','r') as f:
#     x_train = f['input'][...]
#     y_train = f['output'][...]
#     train_dist = f['sensor_dist'][...]
#     #event_num_train = f['event)num'][...]

with h5py.File('../data/lstm_norm_data/long_horizon/final/train/snorkel_train_5.h5','r') as f:
    x_train = f['input'][...]
    y_train = f['output'][...]
    #train_dist = f['sensor_dist'][...]
    #event_num_train = f['event)num'][...]
y_train = np.where(y_train > 0.5, 1, 0)
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

x_val = np.concatenate([x_val])
y_val = np.concatenate([y_val])

profiles_use = [0, 3, 7]
x_train = np.transpose(x_train, axes=[0, 2, 1])[:, -steps:, profiles_use]
x_val = np.transpose(x_val, axes=[0, 2, 1])[:, -steps:, profiles_use]

sclr = MinMax(np.min(x_train, axis=(0, 1)), np.max(x_train, axis=(0, 1)))
x_train = sclr.transform(x_train)
x_val = sclr.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)

svm = SVC(probability=True)
svm.fit(x_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)


def make_prob_pred(x, y):
    f_x = svm.predict_proba(x)[:, 1]
    g_x = knn.predict_proba(x)[:, 1]
    p = np.zeros(y.shape)
    mask = np.where(f_x > 0.6)[0]
    p[mask] = 1
    output_prob = f_x * p + g_x * (1 - p)
    print(np.unique(p, return_counts=True))
    return output_prob


val_pred = make_prob_pred(x_val, y_val)
val_pred = np.where(val_pred > 0.5, 1, 0)
print("val classification report is \n", classification_report(y_val, val_pred))

#fpr, tpr, thresholds = roc_curve(y_val, val_pred)



#gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
#ix = np.argmax(gmeans)
#print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
# plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
# plt.plot(fpr, tpr, marker='.', label='Logistic')
# plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# # axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# # show the plot
# plt.show()

# precision, recall, thresholds = precision_recall_curve(y_val, val_pred)
# # convert to f score
# fscore = (2 * precision * recall) / (precision + recall)
# # locate the index of the largest f score
# ix = np.argmax(fscore)
# print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

# opt_threshold = thresholds[ix]


# start_time = time.time()

# test_pred = make_prob_pred(x_test, y_test)
# test_pred = np.where(test_pred > 0.5 , 1, 0)
# print("--- %s seconds ---" % (time.time() - start_time))
# print("Test classification report is \n", classification_report(y_test, test_pred))
# print("Test confusion matrix is \n", confusion_matrix(y_test, test_pred))

#np.savetxt('../data/temp/free_flow_filtered_knn_output_prob.txt', test_pred)


# output = np.where(test_pred > opt_threshold, 1, 0)

# print("The f1 score is", f1_score(y_test, output))

# num_detected = np.sum(output)
# num_incident = np.sum(y_test)

# num_correct = len(np.where((output + y_test) == 2)[0])

# print('Detection Rate', num_correct/num_incident)
# print('False Alarm Rate', (num_detected - num_correct)/num_detected)


# y_pred_t = test_pred
# y_pred_t = np.where(y_pred_t > opt_threshold, 1, 0)
# free_flow_idx_t = np.where(x_test_org[:, 0, -1] >= 0.85)[0]
# y_pred_t[free_flow_idx_t] = 0
# y_pred_fil_t = group_slices(y_pred_t, event_num_test, 360, 160)



# print('Testing Results')

# scores(y_pred_fil_t, y_test, event_num_test)
