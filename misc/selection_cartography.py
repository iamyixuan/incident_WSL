import glob
import h5py
import numpy as np
import pandas as pd
import yaml
import os


from lstm_cartography import training
from data_map_utils import TrainingDynamics

with h5py.File("../data/lstm_norm_data/long_horizon/train/train.h5", 'r') as g:
    x_train = g["input"][...]
    y_train = g["output"][...]
    event_train = g["event_num"][...]

speed_diff = (x_train[:, 3, :] - x_train[:, 0, :]) / 0.5 * (x_train[:, 3, :] + x_train[:, 0, :])
occ_diff = (x_train[:, 5, :] - x_train[:, 2, :]) / 0.5 * (x_train[:, 5, :] + x_train[:, 2, :])
speed_diff = np.expand_dims(speed_diff, axis=1)
occ_diff = np.expand_dims(occ_diff, axis=1)
print(speed_diff.shape)
x_train = np.concatenate([x_train, speed_diff, occ_diff], axis=1)

all_id = np.arange(x_train.shape[0]) # use to keep track of time slices

with open('./config.yaml') as f:
    config = yaml.load(f)

batch_size = config['hyperparameters']['batch_size']
num_epochs = config['hyperparameters']['num_epochs']
learning_rate = config['hyperparameters']['learning_rate']
cell_type = config['hyperparameters']['cell_type']
if_bidirectional = config['hyperparameters']['if_bidirectional']
num_lstm_layers = config['hyperparameters']['num_lstm_layers']
num_units = config['hyperparameters']['num_units']
num_fc_layers = config['hyperparameters']['num_fc_layers']
num_neurons = config['hyperparameters']['num_neurons']
w = config['hyperparameters']['w']
gamma = config['hyperparameters']['gamma']
patience = config['hyperparameters']['patience']
horizon = config['hyperparameters']['horizon']

start_id = 0
end_id = 361 * 3
count = 0
while True:
    dir_name = "./cartography_train_easy/training_dynamics_" + str(count)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if end_id == 361 * 3:
        x = x_train[start_id : end_id]
        y = y_train[start_id : end_id]
        slice_id = all_id[start_id : end_id]
    else:
        x = np.append(x_train[start_id : end_id], sample_keep_x, axis=0)
        y = np.append(y_train[start_id : end_id], sample_keep_y, axis=0)
        slice_id = np.append(all_id[start_id : end_id], sample_keep_id, axis=0)

    print("current iteration {} and data size {}".format(count, x.shape[0]))
    training(x_train=x,
            y_train=y,
            dir_name=dir_name,
            batch_size=batch_size,
            num_epochs = num_epochs,
            lr=learning_rate,
            cell_type = cell_type,
            if_bidirectional=if_bidirectional,
            num_lstm_layers=num_lstm_layers,
            num_units=num_units,
            num_fc_layers=num_fc_layers,
            num_neurons=num_neurons,
            w=w,
            gamma=gamma,
            patience=patience,
            horizon=horizon)
    
    files = glob.glob(dir_name + "/*.csv")
    pred_mat = []
    true_mat = []
    ids = []
    for f in files:
        tmp_df = pd.read_csv(f)
        pred_mat.append(tmp_df["0"])
        true_mat.append(tmp_df["y_true"])
        ids.append(tmp_df["id"])

    true = np.array(true_mat).T
    pred = np.array(pred_mat).T

    training_dynamics = TrainingDynamics(true, pred)
    confidence = training_dynamics.get_confidence()
    variability = training_dynamics.get_variability()
    correctness = training_dynamics.get_correntness()
    correctness = correctness/num_epochs

    sample_keep_idx = np.where(correctness > 0.7)[0]
    ids = ids[0].values[sample_keep_idx]

    sample_keep_x = x[ids]
    sample_keep_y = y[ids]
    sample_keep_id = slice_id[ids]

    start_id = sample_keep_id.max()
    end_id = int(0.3 * len(sample_keep_id)) + start_id # adding 50% new data points each iteration
    count += 1

    print("current selected sample size is ", len(sample_keep_x))
    if end_id >= all_id.max():
        break








# for i, event in enumerate(np.unique(event_train)):
#     idx = np.where(event_train == event)[0]
#     dir_name = "./training_dynamics_" + str(i)
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     if i == 0:
#         x = x_train[idx]
#         y = y_train[idx]
#         slice_id = all_id[idx]
#     else:
#         x = np.append(x_train[idx], sample_keep_x, axis=0)
#         y = np.append(y_train[idx], sample_keep_y, axis=0)
#         slice_id = np.append(all_id[idx], sample_keep_id, axis=0)

#     print("current iteration {} and data size {}".format(i, x.shape[0]))
#     training(x_train=x,
#             y_train=y,
#             dir_name=dir_name,
#             batch_size=batch_size,
#             num_epochs = num_epochs,
#             lr=learning_rate,
#             cell_type = cell_type,
#             if_bidirectional=if_bidirectional,
#             num_lstm_layers=num_lstm_layers,
#             num_units=num_units,
#             num_fc_layers=num_fc_layers,
#             num_neurons=num_neurons,
#             w=w,
#             gamma=gamma,
#             patience=patience,
#             horizon=horizon)
    
#     files = glob.glob(dir_name + "/*.csv")
#     pred_mat = []
#     true_mat = []
#     ids = []
#     for f in files:
#         tmp_df = pd.read_csv(f)
#         pred_mat.append(tmp_df["0"])
#         true_mat.append(tmp_df["y_true"])
#         ids.append(tmp_df["id"])

#     true = np.array(true_mat).T
#     pred = np.array(pred_mat).T

#     training_dynamics = TrainingDynamics(true, pred)
#     confidence = training_dynamics.get_confidence()
#     variability = training_dynamics.get_variability()
#     correctness = training_dynamics.get_correntness()
#     correctness = correctness/num_epochs

#     sample_keep_idx = np.where(correctness > 0.25)[0]
#     ids = ids[0].values[sample_keep_idx]

#     sample_keep_x = x[ids]
#     sample_keep_y = y[ids]
#     sample_keep_id = slice_id[ids]

#     print("current selected sample size is ", len(sample_keep_x))

np.savetxt("./cartography_train_easy/easy_slices_ids_right_label_easy.txt", sample_keep_id)
with h5py.File("../data/lstm_norm_data/long_horizon/selected_train_from_original_train_easy.h5", "w") as f:
    f.create_dataset("input", data=sample_keep_x)
    f.create_dataset("output", data=sample_keep_y)
