import numpy as np
import tensorflow as tf
import time
import pickle
import glob
import os
import h5py
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


np.random.seed(1)
tf.random.set_seed(1)

def add_diff(x):
    speed_diff = (x[:, 3, :] - x[:, 0, :]) / 0.5 * (x[:, 3, :] + x[:, 0, :])
    occ_diff = (x[:, 5, :] - x[:, 2, :]) / 0.5 * (x[:, 5, :] + x[:, 2, :])
    speed_diff = np.expand_dims(speed_diff, axis=1)
    occ_diff = np.expand_dims(occ_diff, axis=1)
    x = np.concatenate([x, speed_diff, occ_diff], axis=1)
    return x
   


def inference(scaler_name, model_name, x_val, x_test):

    horizon = 15
    steps = int(horizon * 60 / 30)

    profiles_use = [0, 3, 7]

    x_val = np.transpose(x_val, axes=[0, 2, 1])[:, -steps:, profiles_use]
    x_test = np.transpose(x_test, axes=[0, 2, 1])[:, -steps:, profiles_use]

    sclr = pickle.load(open(scaler_name, "rb"))

    x_val = sclr.transform(x_val)
    x_test = sclr.transform(x_test)

    model = tf.keras.models.load_model(model_name)

    y_pred_val = model.predict(x_val)
    y_pred_test = model.predict(x_test)

    return  y_pred_val, y_pred_test


if __name__ == "__main__":
    scaler_names = sorted(glob.glob("../saved_models/scalers/*.pkl"))
    model_names = sorted(glob.glob("../saved_models/models/*.h5"))

    with h5py.File("../data/lstm_norm_data/long_horizon/hand_pick.h5", "r") as f:
        x_train_1 = f["x_train"][...]
        y_train_1 = f["y_train"][...]
        x_val_1 = f["x_val"][...]
        y_val_1 = f["y_val"][...]
    with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/val/pickedVal.h5','r') as f:
        x_val = f['input'][...]
        y_val = f['output'][...]

    with open("../data/lstm_norm_data/long_horizon/final/testTimeDistance/testWithTimeDistance_offset.pkl", "rb") as g:
        test_data = pickle.load(g)

    x_val = add_diff(x_val)
    x_val = np.concatenate([x_val_1, x_val])
    y_val = np.concatenate([y_val_1, y_val])
    x_test = test_data["input"].astype(np.float32)
    x_test = add_diff(x_test)


    val_pred = []
    test_pred = []
    for scaler_name, model_name in zip(scaler_names, model_names):
        print(model_name)
        print(scaler_name)
        val_tmp, test_tmp = inference(scaler_name, model_name, x_val, x_test)
        val_p_tmp = np.where(val_tmp > 0.5, 1, 0)
        val_acc = accuracy_score(y_val, val_p_tmp)
        print("Validation acc is %.3f" % val_acc)
        if val_acc > 0.90:
            val_pred.append(val_tmp)
            test_pred.append(test_tmp)
        else:
            os.remove(scaler_name)
            os.remove(model_name)
            pass


    print("Numer of models in the ensemble is %d" % len(val_pred))
    # np.savetxt("../data/temp/val_pred_mean_ensemble.txt",np.mean(val_pred, axis=0))
    # np.savetxt("../data/temp/val_pred_std_ensemble.txt",np.std(val_pred, axis=0))
    # np.savetxt("../data/temp/test_pred_mean_ensemble.txt",np.mean(test_pred, axis=0))
    # np.savetxt("../data/temp/test_pred_std_ensemble.txt",np.std(test_pred, axis=0))
    np.savetxt("../data/temp/test_pred_all_ensemble.txt",np.squeeze(test_pred))
    np.savetxt("../data/temp/val_pred_all_ensemble.txt", np.squeeze(np.array(val_pred)))




    
    
    

    # start_time = time.time()
    # y_pred_t = model.predict(x_test)
    # label_pred = np.where(y_pred_t > 0.5, 1, 0)
    # free_flow = np.where((x_test[:, -1, 0] >= 0.8) & (x_test[:, -1, 1] >= 0.8))[0]
    # label_pred[free_flow] = 0
    # #np.savetxt("../data/temp/test_pred_NoDistModel_TimeDistance_combined_offsetDoubleDrop.txt", y_pred_t)
    # print(classification_report(y_test, label_pred))
    # print("confusion matrix \n", confusion_matrix(y_test, label_pred))

    label_pred_val = np.where(np.array(val_pred).mean(axis=0) > 0.5, 1, 0)
    free_flow_val = np.where((x_val[:, -1, 0] >= 0.8) & (x_val[:, -1, 1] >= 0.8))[0]
    label_pred_val[free_flow_val] = 0
    #np.savetxt("../data/temp/val_pred_NoDistModel_TimeDistanceDoubleDrop", y_pred_val)
    print(classification_report(y_val, label_pred_val))
    print("confusion matrix \n", confusion_matrix(y_val, label_pred_val))



    """
    save prediction indices
    """
    # pos_idx = np.where(label_pred == 1)[0]
    # neg_idx = np.where(label_pred == 0)[0]

    # np.savetxt("../data/temp/posPredIdx_TimeDistanceVal.txt", pos_idx)
    # np.savetxt("../data/temp/negPredIdxTimeDistanceVal.txt", neg_idx)

    # detections = get_detection(y_pred_t, x_test, event_num_test, 0, "lol")
    # detections = np.concatenate(detections).reshape(-1, 1)
    # print(scores(detections, y_test, event_num_test))

    # x_test = tf.Variable(x_val, dtype=float)
    # with tf.GradientTape() as tape:
    #     pred = model(x_test, training=False)
    # grads = tape.gradient(pred, x_test)
    # #np.save("../data/saved_predictions/saliency_mapValCombinedDoubleDrop.npy", grads)
    # print(grads.shape)
    # print(y_pred_val.max(), y_pred_val.min())
    # print("--- %s seconds ---" %(time.time() - start_time))
    

    # precision, recall, thresholds = precision_recall_curve(y_test, y_pred_t)
    # print(precision, recall)
    # # convert to f score
    # fscore = (2 * precision * recall) / (precision + recall + 1e-6)
    # print(fscore)
    # # locate the index of the largest f score
    # ix = np.argmax(fscore)
    # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

    #np.savetxt('../data/saved_predictions/model_on_selected_data_test_speed.txt', y_pred_t)




    
"""
    y_pred_t = np.where(y_pred_t > threshold, 1, 0)
    np.savetxt('../data/saved_predictions/rnn_pred_raw.txt', y_pred_t)

    num_detected = np.sum(y_pred_t)
    num_incident = np.sum(y_test)
    num_correct = len(np.where((y_pred_t + y_test) == 2)[0])

    print('Detection Rate', num_correct/num_incident)
    print('FAR', (num_detected - num_correct)/num_detected)
    print('f1', f1_score(y_test, y_pred_t))

    free_flow_idx_t = np.where(x_test_org[:, 0, -1] >= 0.85)[0]
    y_pred_t[free_flow_idx_t] = 0
    y_pred_fil_t = group_slices(y_pred_t, event_num_test, 360, 160)

    # np.savetxt('../data/saved_predictions/y_val_2019_Mar_Apr.txt', y_val)
    # np.savetxt('../data/saved_predictions/y_val_pred_2019_Mar_Apr.txt', y_pred_fil)
    print('\n Testing Results')

    scores(y_pred_fil_t, y_test, event_num_test)

# # np.savetxt('../data/saved_predictions/y_test_2019_12.txt', y_test)
# # np.savetxt('../data/saved_predictions/y_test_pred_2019_12.txt', y_pred_fil_t)
"""