import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import glob
import h5py
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# load model predictions


class TrainingDynamics:
    def __init__(self, true, pred):
        self.true = true
        self.pred = pred

    def get_confidence(self):
        """
        pred should have shape (sample size, num epochs)
        """
        p_t = self.pred * self.true + (1 - self.pred) * (1 - self.true)
        return np.mean(p_t, axis=1)

    def get_variability(self):
        p_t = self.pred * self.true + (1 - self.pred) * (1 - self.true)
        return np.std(p_t, axis=1)

    def get_correntness(self):
        """
        take 0.5 as the classification threshold.
        """
        correctness = np.zeros(self.true.shape[0])
        for i in range(self.true.shape[0]):
            tmp_pred_vec = np.where(self.pred[i] >= 0.5, 1, 0)
            counts = len(np.where(tmp_pred_vec == self.true[i])[0])
            correctness[i] = counts
        return correctness


files = glob.glob("./training_dynamics_16/*.csv")
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
print(true.shape)




training_dynamics = TrainingDynamics(true, pred)
confidence = training_dynamics.get_confidence()
variability = training_dynamics.get_variability()
correctness = training_dynamics.get_correntness()
correctness = correctness/correctness.max()
print(correctness[:5])
hard_to_learn = np.where(correctness < 0.3)[0]
ambiguous = np.where((correctness>=0.3) & (correctness < 0.7))[0]
easy_to_learn = np.where(correctness >= 0.7)[0]
# ids = ids[0].values[hard_to_learn]

harder = np.where(correctness < 0.5)[0]


fig, ax = plt.subplots(figsize=(6, 5))

im = ax.scatter(variability, confidence, c=correctness)
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 1)
ax.set_xlabel("variability")
ax.set_ylabel("confidence")
#ax.axhline(0.5, color='r')
cbar = plt.colorbar(im)
cbar.set_label("correctness")
plt.show()



# with h5py.File("../data/lstm_norm_data/long_horizon/final/val/val.h5", 'r') as g:
#     x_train = g["input"][...]
#     y_train = g["output"][...]
#     event_train = g["event_num"][...]

# easy_id = np.loadtxt("../data/temp/easy_slices_ids_right_label_easy.txt")
# speed_diff = (x_train[:, 3, :] - x_train[:, 0, :]) / 0.5  * (x_train[:, 3, :] + x_train[:, 0, :])
# occ_diff = (x_train[:, 5, :] - x_train[:, 2, :]) / 0.5 * (x_train[:, 5, :] + x_train[:, 2, :])
# print("hehre", speed_diff[0:2, -1])
# speed_diff = np.expand_dims(speed_diff, axis=1)
# print("lol", speed_diff[0:2, :, -1])
# occ_diff = np.expand_dims(occ_diff, axis=1)
# x_train = np.concatenate([x_train, speed_diff, occ_diff], axis=1)
# print(x_train.shape)
# print("lol", x_train[0:2, 6, -1])

# ids = np.arange(x_train.shape[0])
# # id_easy = ids[easy_to_learn]
# # id_amb = ids[ambiguous]
# # id_hard = ids[hard_to_learn]

# for i, d in enumerate(np.unique(event_train)):
#     idx_ = np.where(event_train == d)[0]
#     id_current = ids[idx_]  
#     label = y_train[idx_]
#     incident = np.where(label == 1)[0]
#     fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
#     ax[0].plot(x_train[idx_, 0, -1])
#     ax[0].plot(x_train[idx_, 3, -1])
#     ax[0].set_title("Speed")
    
#     ax[2].set_title("Occupancy")
#     ax[1].set_title("Relative Speed Difference")
#     ax[3].set_title("Relative Occupancy Difference")

#     ax[2].plot(x_train[idx_, 2, -1])
#     ax[2].plot(x_train[idx_, 5, -1])
#     ax[1].plot(x_train[idx_, 7, -1])
#     ax[3].plot(x_train[idx_, 8, -1])
#     #ax[0].axvspan(incident[0], incident[-1], color='r', alpha=0.1)
#     ax[1].axvspan(incident[0], incident[-1], color='r', alpha=0.1)
    
#     ax[2].axvspan(incident[0], incident[-1], color='r', alpha=0.1)
#     ax[3].axvspan(incident[0], incident[-1], color='r', alpha=0.1)

#     # for a in ax:
#     #     a.axis(False)
#     for j, id in enumerate(id_current):
#         if not id in easy_id:
#             ax[0].axvline(j, color='purple', alpha=0.2)

#         # if id in id_easy:
#         #     ax[0].axvline(j, color='green', alpha=0.2)
#         # elif id in id_amb:
#         #     ax[0].axvline(j, color='blue', alpha=0.2)
#         # elif id in id_hard:
#         #     ax[0].axvline(j, color='red', alpha=0.2)
#     fig_dir = "../notes/plots/train_map_human_detectable_harder/right_label_res_val_easy0.7/"
#     if not os.path.exists(fig_dir):
#         os.makedirs(fig_dir)
#     fig.savefig( fig_dir + str(i) + ".png", format="png", bbox_inches="tight")
#     plt.close()
