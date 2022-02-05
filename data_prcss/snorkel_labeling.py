import sys
import numpy as np
import pickle 
import glob
import h5py
import tensorflow as tf

from snorkel.labeling import labeling_function, LFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

np.random.seed(0)

def create_training_data(
    speed_sep_low,
    speed_sep_high,
    speed_sep_low_harsh,
    speed_sep_high_harsh,
    occ_sep_low,
    occ_sep_high,
    occ_sep_low_harsh,
    occ_sep_high_harsh,
    speed_single_low,
    speed_single_high,
    occ_single_low,
    occ_single_high,
    speed_double_high,
    occ_double_high,
    epochs,
    step_size,
    decay
):
    INCIDENT = 1
    NONINCIENT = 0
    ABSTAIN = -1

    @labeling_function()
    def check_speed_sep(x):
        """
        x: input concatenate sequence which has the shape (seq_len, 7)
        """
        up_speed = x[0, -20:]
        down_speed = x[3, -20:]
        relative_diff = (down_speed - up_speed)/ (0.5 * (down_speed + up_speed))

        if  np.all(np.abs(relative_diff) >= speed_sep_high):
            return INCIDENT
        elif np.all(np.abs(relative_diff) <= speed_sep_low):
            return NONINCIENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_speed_sep_2(x):
        """
        x: input concatenate sequence which has the shape (seq_len, 7)
        """
        up_speed = x[0, -20:]
        down_speed = x[3, -20:]
        relative_diff = (down_speed - up_speed)/ (0.5 * (down_speed + up_speed))

        if  np.all(np.abs(relative_diff) >= speed_sep_high_harsh):
            return INCIDENT
        elif np.all(np.abs(relative_diff) <= speed_sep_low_harsh):
            return NONINCIENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_occ_sep(x):
        """
        x: input concatenate sequence which has the shape (seq_len, 7)
        """
        up_speed = x[2, -20:]
        down_speed = x[5, -20:]
        relative_diff = (down_speed - up_speed) / (0.5 * (down_speed + up_speed))

        if  np.all(np.abs(relative_diff) >= occ_sep_high):
            return INCIDENT
        elif np.all(np.abs(relative_diff) <= occ_sep_low):
            return NONINCIENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_occ_sep_2(x):
        """
        x: input concatenate sequence which has the shape (seq_len, 7)
        """
        up_occ = x[2, -20:]
        down_occ = x[5, -20:]
        relative_diff = (down_occ - up_occ) / (0.5 * (down_occ + up_occ))

        if  np.all(np.abs(relative_diff) >= occ_sep_high_harsh):
            return INCIDENT
        elif np.all(np.abs(relative_diff) <= occ_sep_low_harsh):
            return NONINCIENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_freeflow(x):
        """
        checking if both upstream and downstream speed are over 0.8
        only need to check the short period of time at the end (3min)
        """
        up_speed = x[0, -6:]
        down_speed = x[3, -6:]
        if np.all(up_speed >= 0.8) and np.all(down_speed >= 0.8):
            return NONINCIENT
        else:
            return ABSTAIN


    @labeling_function()
    def check_single_speed_drop(x):
        """
        check end point speed within one sequence
        """
        start_upspeed = x[0, -20]
        end_upspeed = x[3, -20]
        up_diff = start_upspeed - end_upspeed

        start_downspeed = x[0, -1]
        end_downspeed = x[3, -1]
        down_diff = start_downspeed - end_downspeed

        if np.abs(up_diff) < speed_single_low and np.abs(down_diff) > speed_single_high:
            return INCIDENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_single_occ_increase(x):
        """
        check end point speed within one sequence
        """
        start_upOcc = x[2, -20]
        end_upOcc = x[5, -20]
        up_diff = start_upOcc - end_upOcc

        start_downOcc = x[2, -1]
        end_downOcc = x[5, -1]
        down_diff = start_downOcc - end_downOcc

        if np.abs(up_diff) < occ_single_low and np.abs(down_diff) > occ_single_high:
            return INCIDENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_double_speed_drop(x):
        """
        check if one speed is constant while the other going down
        """
        start_upspeed = x[0, -20]
        end_upspeed = x[0, -1]
        up_diff = start_upspeed - end_upspeed

        start_downspeed = x[3, -20]
        end_downspeed = x[3, -1]
        down_diff = start_downspeed - end_downspeed

        if up_diff > speed_double_high and down_diff > speed_double_high:
            return INCIDENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_double_occ_increase(x):
        """
        check if one speed is constant while the other going down
        """
        start_upOcc = x[2, -20]
        end_upOcc = x[2, -1]
        up_diff = end_upOcc - start_upOcc

        start_downOcc = x[5, -20]
        end_downOcc = x[5, -1]
        down_diff = end_downOcc - start_downOcc

        if up_diff > occ_double_high and down_diff > occ_double_high:
            return INCIDENT
        else:
            return ABSTAIN

    @labeling_function()
    def check_if_flat(x):
        """
        if any speed or occ profile is flat do not label this point
        """
        x = x[(0, 2, 3, 5), -20:]
        if np.any(np.std(x, axis=1) == 0):
            return NONINCIENT
        else:
            return ABSTAIN
        
    with open("../data/lstm_norm_data/long_horizon/train/TrainWithTimeDistance.pkl", "rb") as f:
        data = pickle.load(f)
    inputs = data["input"]

    test_input = inputs
    lfs = [check_speed_sep, check_single_speed_drop, check_double_speed_drop, 
            check_occ_sep, check_speed_sep_2, check_occ_sep_2, check_single_occ_increase, check_double_occ_increase,
            check_freeflow, check_if_flat]
    applier = LFApplier(lfs=lfs)
    L_train = applier.apply(test_input)


    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=epochs, step_size=step_size, decay=decay, seed=0)

    pred_prob = label_model.predict_proba(L_train)
    prob_label_tmp = pred_prob[:, 1]
    data_left = np.where(prob_label_tmp!=0.5)[0]
    prob_label = prob_label_tmp[data_left]

    train_data = {"x":  test_input[data_left], "y": prob_label}
    return train_data