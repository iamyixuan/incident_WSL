import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import re

np.random.seed(0)

mpl.rcParams["font.size"] = "14"
#mpl.rcParams["font.weight"] = "bold"
fontsize = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.edgecolor'] = 'black'



with open("../data/lstm_norm_data/long_horizon/final/testTimeDistance/val/ValWithTimeDistance.pkl", "rb") as file:
    data = pickle.load(file)

labels = data["output"]
x = data["input"]
y = data["output"]
dist = data["sensor_dist"]

# pos_idx = np.where(y == 1)[0]
# neg_idx = np.where(y == 0)[0]
# np.random.shuffle(pos_idx)
# np.random.shuffle(neg_idx)
# # x_train = np.concatenate([x[pos_idx], x[neg_idx]])
# # y_train = np.concatenate([y[pos_idx], y[neg_idx]])
# # dist = np.concatenate([dist[pos_idx], dist[neg_idx]])
# # print(x_train.shape)
# # with h5py.File("../data/lstm_norm_data/long_horizon/final/testTimeDistance/train/pickedTrain.h5", "w") as f:
# #     f.create_dataset("input", data=np.array(x_train))
# #     f.create_dataset("output", data=np.array(y_train))
# #     f.create_dataset("sensor_dist", data=np.array(dist))

# for i, idx in enumerate(neg_idx):
#     idx = int(idx)
#     up = x[idx, 0, -40:]
#     down = x[idx, 3, -40:]
#     up_occ = x[idx, 2, -40:]
#     down_occ = x[idx, 5, -40:]
#     diff = np.abs((down - up) / (0.5 * (down + up)))
#     diff_occ = np.abs((down_occ - up_occ) / (0.5 * (down_occ + up_occ)))
    
#     fig, ax = plt.subplots(2, 3, figsize=(10, 5))
#     for j in range(2):
#         ax[j][0].plot(up if j==0 else up_occ, linewidth=2, label=('Up%s' % ("Speed" if j==0 else "Occ")))
#         ax[j][0].set_ylim(-0.2, 1.2 if j==0 else 2)
#         ax[j][1].plot(down if j==0 else down_occ, label=("DownSpeed %s" % ("Speed" if j==0 else "Occ")))
#         ax[j][1].set_ylim(-0.2, 1.2 if j==0 else 2)
#         ax[j][2].plot(diff if j==0 else diff_occ, label=("Rel Diff %s" % ("Speed" if j==0 else "Occ")))
#         ax[j][2].set_ylim(0, 1.2)
#     for ax_l in ax.reshape(-1, ):
#         ax_l.legend(fontsize=10)
#     #fig.suptitle("Sensor Dist is %.3f" % dist[idx])
#     fig.savefig("../notes/plots/TestWithTimeDist/val_neg/%s.png" % str(idx), format="png", dpi=100, bbox_inches="tight")
#     plt.close()
#     if i >=1000:
#         break


"""
Generate datasets based on picked indices
"""

# x = data["input"]
# y = data["output"]
# dist = data["sensor_dist"]


file_idx_pos = os.listdir("../notes/plots/TestWithTimeDist/val_pos_pick/")
file_idx_neg = os.listdir("../notes/plots/TestWithTimeDist/val_neg_pick/")

idxs_pos = []
for f in file_idx_pos:
    if f != ".DS_Store":
        tmp = re.findall("[0-9]+", f)
        idxs_pos.append(int(tmp[0]))

idxs_neg = []
for g in file_idx_neg:
    if g != ".DS_Store":
        tmp_n = re.findall("[0-9]+", g)
        idxs_neg.append(int(tmp_n[0]))

idxs = np.array(idxs_pos + idxs_neg)

with h5py.File("../data/lstm_norm_data/long_horizon/final/testTimeDistance/val/val_for_snorkel.h5", "w") as f:
    f.create_dataset("input", data=x[idxs])
    f.create_dataset("output", data=y[idxs])
    f.create_dataset("sensor_dist", data=dist[idxs])

# for i, idx in enumerate(idxs_pos):
#     idx = int(idx)
#     up = x[idx, 2, -20:]
#     down = x[idx, 5, -20:]
#     diff = (down - up) / (0.5 * (down + up))
#     #print(up[:5], down[:5], diff[:5])
#     fig, ax = plt.subplots(1, 3, figsize=(15, 3))
#     ax[0].plot(up, linewidth=2, label='UpSpeed')
#     ax[0].set_ylim(-0.5, 4)
#     ax[1].plot(down, label="DownSpeed")
#     ax[1].set_ylim(-0.5, 4)
#     ax[2].plot(diff, label="Rel Diff")
#     ax[2].set_ylim(-3, 4)
#     #fig.suptitle("Sensor Dist is %.3f" % dist[idx])
#     fig.savefig("../notes/plots/TestWithTimeDist/train_pos_occ/%s.png" % str(idx), format="png", dpi=100, bbox_inches="tight")
#     plt.close()
    # if i >=2:
    #     break
