import numpy as np
import h5py
import pandas as pd

train_pos_idx = pd.read_csv("../notes/week_6_7/hand_pick_train_pos.txt", header=None).values.reshape(-1, )
train_neg_idx = pd.read_csv("../notes/week_6_7/hand_pick_train_neg.txt", header=None).values.reshape(-1, )

val_pos_idx = pd.read_csv("../notes/week_6_7/hand_pick_val_pos.txt", header=None).values.reshape(-1, )
val_neg_idx = pd.read_csv("../notes/week_6_7/hand_pick_val_neg.txt", header=None).values.reshape(-1, )

with h5py.File("../data/lstm_norm_data/long_horizon/selected_train_from_original.h5", "r") as train:
    x_train = train["input"][...]
    y_train = train["output"][...]

with h5py.File("../data/lstm_norm_data/long_horizon/selected_val_from_original.h5", "r") as val:
    x_val = val["input"][...]
    y_val = val["output"][...]


train_pos_all = np.where(y_train==1)[0]
train_neg_all = np.where(y_train==0)[0]
val_pos_all = np.where(y_val==1)[0]
val_neg_all = np.where(y_val==0)[0]


train_save_idx = np.concatenate([train_pos_all[train_pos_idx], train_neg_all[train_neg_idx]])

x_train_save = x_train[train_save_idx]
y_train_save = y_train[train_save_idx]


val_save_idx = np.concatenate([val_pos_all[val_pos_idx], val_neg_all[val_neg_idx]])
x_val_save = x_val[val_save_idx]
y_val_save = y_val[val_save_idx]

with h5py.File("../data/lstm_norm_data/long_horizon/hand_pick.h5", "w") as f:
    f.create_dataset("x_train", data=x_train_save)
    f.create_dataset("y_train", data=y_train_save)
    f.create_dataset("x_val", data=x_val_save)
    f.create_dataset("y_val", data=y_val_save)