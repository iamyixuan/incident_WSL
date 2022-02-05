import numpy as np
import tensorflow as tf
import time
import pickle
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, confusion_matrix

from utils import MinMax, scores, get_detection
import h5py

np.random.seed(1)
tf.random.set_seed(1)

def add_diff(x):
    speed_diff = (x[:, 3, :] - x[:, 0, :]) / 0.5 * (x[:, 3, :] + x[:, 0, :])
    occ_diff = (x[:, 5, :] - x[:, 2, :]) / 0.5 * (x[:, 5, :] + x[:, 2, :])
    speed_diff = np.expand_dims(speed_diff, axis=1)
    occ_diff = np.expand_dims(occ_diff, axis=1)
    x = np.concatenate([x, speed_diff, occ_diff], axis=1)
    return x

    



def training(i):
    # with h5py.File('../data/lstm_norm_data/long_horizon/selected_train_from_original.h5','r') as f:
    #     x_train = f['input'][...]
    #     y_train = f['output'][...]
    #     #event_num_train = f['event_num'][...]

    # with h5py.File('../data/lstm_norm_data/long_horizon/val/val.h5','r') as f:
    #     x_val = f['input'][...]
    #     y_val = f['output'][...]
    #     event_num_val = f['event_num'][...]

    # with h5py.File("../data/lstm_norm_data/long_horizon/hand_pick.h5", "r") as f:
    #     x_train = f["x_train"][...]
    #     y_train = f["y_train"][...]
    #     x_val = f["x_val"][...]
    #     y_val = f["y_val"][...]

    with h5py.File("../data/lstm_norm_data/long_horizon/hand_pick.h5", "r") as f:
        x_train_1 = f["x_train"][...]
        y_train_1 = f["y_train"][...]
        x_val_1 = f["x_val"][...]
        y_val_1 = f["y_val"][...]

    with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/train/pickedTrain.h5','r') as f:
        x_train = f['input'][...]
        y_train = f['output'][...]
        train_dist = f['sensor_dist'][...]
        #event_num_train = f['event)num'][...]

    with h5py.File('../data/lstm_norm_data/long_horizon/final/testTimeDistance/val/pickedVal.h5','r') as f:
        x_val = f['input'][...]
        y_val = f['output'][...]
        val_dist = f['sensor_dist'][...]
        #event_num_val = f['event)num'][...]
    horizon=10
    steps = int(horizon * 60 / 30)

    def add_diff(x):
        speed_diff = (x[:, 3, :] - x[:, 0, :]) / (0.5  * (x[:, 3, :] + x[:, 0, :]))
        occ_diff = (x[:, 5, :] - x[:, 2, :]) / (0.5 * (x[:, 5, :] + x[:, 2, :]))
        speed_diff = np.expand_dims(speed_diff, axis=1)
        occ_diff = np.expand_dims(occ_diff, axis=1)
        x = np.concatenate([x, speed_diff, occ_diff], axis=1)
        return x
    
    x_train = add_diff(x_train)
    x_val = add_diff(x_val)

    x_train = np.concatenate([x_train_1, x_train])
    y_train = np.concatenate([y_train_1, y_train])

    x_val = np.concatenate([x_val_1, x_val])
    y_val = np.concatenate([y_val_1, y_val])


    with open("../data/lstm_norm_data/long_horizon/final/testTimeDistance/testWithTimeDistance_offset.pkl", "rb") as g:
        test_data = pickle.load(g)
    
    x_test = test_data["input"].astype(np.float32)
    y_test = test_data["output"].astype(np.float32)


    x_train_org = x_train
    x_val_org = x_val
    x_test_org = 0


    # x_test = x_val.astype(np.float32)
    # y_test = y_val.astype(np.float32)

    x_train = add_diff(x_train)
    x_val = add_diff(x_val)

    x_test = add_diff(x_test)

    steps = int(10 * 60 / 30)

    profiles_use = [0, 2, 3, 5]
    x_train = np.transpose(x_train, axes=[0, 2, 1])[:, -steps:, profiles_use]
    x_val = np.transpose(x_val, axes=[0, 2, 1])[:, -steps:, profiles_use]
    x_test = np.transpose(x_test, axes=[0, 2, 1])[:, -steps:, profiles_use]

    
    sclr = pickle.load(open("../saved_models/model_snorkel_5-3.pkl", "rb"))
    x_train = sclr.transform(x_train)
    x_val = sclr.transform(x_val)
    x_test = sclr.transform(x_test)


    y_train = y_train.reshape(-1, 1).astype(np.float32)
    print(np.unique(y_train, return_counts=True))
    y_val = y_val.reshape(-1, 1).astype(np.float32)
    y_test = y_test.reshape(-1, 1).astype(np.float32)



    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))


    def weighted_bce(labels, preds, weight=1.5, gamma=2):
        fl = -(labels * weight * K.pow((1 - preds), gamma) * K.log(preds) + (1 - labels)*K.pow(preds, gamma)*K.log(1 - preds))
        return K.mean(fl)

    def create_model():
        layer = tf.keras.layers.LSTM
        inputs = tf.keras.Input(shape=(5, 6))
        lstm1 = tf.keras.layers.Bidirectional(layer(3))(inputs)
        output = tf.keras.layers.Dense(25, activation='relu')(lstm1)
        output = tf.keras.layers.Dense(10, activation='relu')(output)
        output = tf.keras.layers.Dense(10, activation='relu')(output)
        output = tf.keras.layers.Dense(10, activation='relu')(output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0005, clipvalue=0.5), loss=[weighted_bce], metrics='AUC')
        return model

    # model = create_model()
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1_m', patience=20, mode='max')
    # model.fit(x_train, y_train, epochs=300, verbose=1, validation_data=(x_val, y_val), callbacks=[callback])
    # model.save('../saved_models/lstm_model_pre3'+str(i)+'.h5')


    # y_pred = model.predict(x_val)
    # y_pred = np.where(y_pred > 0.4, 1, 0)
    # print('\n Validation Results (Pre-grouping)')
    # print(classification_report(y_val, y_pred))


    # free_flow_idx_t = np.where(x_val_org[:, 0, -1] >= 0.85)[0]
    # y_pred[free_flow_idx_t] = 0
    # y_pred_val = group_slices(y_pred, event_num_val, 360, 160)

    # scores(y_pred_val, y_val, event_num_val)

    return x_test, y_test, x_val, y_val


# for i in range(1, 10):
#     training(i)

if __name__ == "__main__":
    def weighted_bce(labels, preds, weight=0.5, gamma=2):
        fl = -(labels * weight * K.pow((1 - preds), gamma) * K.log(preds) + (1 - labels)*K.pow(preds, gamma)*K.log(1 - preds))
        return K.mean(fl)
    def focal(labels, preds, weight=0.5, gamma=.5):
        fl = -(labels * weight * K.pow((1 - preds), gamma) * K.log(preds) + (1 - labels)*K.pow(preds, gamma)*K.log(1 - preds))
        return K.mean(fl)
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
    x_test, y_test, x_val, y_val = training(0)

    model = tf.keras.models.load_model("../saved_models/model_snorkel_5-3.h5")
    
    # threshold = 0.44

    start_time = time.time()
    y_pred_t = model.predict(x_test)
    label_pred = np.where(y_pred_t > 0.5, 1, 0)
    free_flow = np.where((x_test[:, -1, 0] >= 0.8) & (x_test[:, -1, 1] >= 0.8))[0]
    label_pred[free_flow] = 0
    np.savetxt("../data/temp/test_snorkel_speed_occ.txt", y_pred_t)
    print(classification_report(y_test, label_pred))
    print("confusion matrix \n", confusion_matrix(y_test, label_pred))


    y_pred_val = model.predict(x_val)
    label_pred_val = np.where(y_pred_val > 0.5, 1, 0)
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