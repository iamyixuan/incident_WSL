import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import MinMax, scores, group_slices
from sklearn.model_selection import train_test_split
import h5py
import yaml
import matplotlib.pyplot as plt




def training(x_train, y_train, dir_name, batch_size, num_epochs, lr, cell_type, if_bidirectional, num_lstm_layers, num_units, num_fc_layers, num_neurons, w=1.5, gamma=2,  patience=10, horizon=20):
    np.random.seed(0)
    tf.random.set_seed(0)


    class Metrics(tf.keras.callbacks.Callback): 
        def __init__(self, val_data, y_true, ids, dir_name):
            super().__init__()
            self.validation_data = val_data
            self.y_true = y_true
            self.id = ids
            self.dir_name = dir_name
                
        def on_epoch_end(self, epoch, logs={}):
            y_pred = self.model.predict(self.validation_data)
            pd.DataFrame(y_pred).assign(epoch=epoch, y_true=self.y_true, id=self.id).to_csv('{}_{}.csv'.format(self.dir_name + "/training_epoch_pred", epoch))




    steps = int(horizon * 60 / 30)

    # x_data = np.concatenate([x_train], axis=0)
    # y_data = np.concatenate([y_train], axis=0)
    train_id = np.arange(0, x_train.shape[0])


    profiles_use = (0, 3, 7)
    x_train = np.transpose(x_train, axes=[0, 2, 1])[:, -steps:, profiles_use]

    sclr = MinMax(np.min(x_train, axis=(0, 1)), np.max(x_train, axis=(0, 1)))
    x_train = sclr.transform(x_train)

    y_train = y_train.reshape(-1, 1).astype(np.float32)

    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    train_id = train_id[shuffle_idx]


    def focal(labels, preds, weight=w, gamma=gamma):
        fl = -(labels * weight * K.pow((1 - preds), gamma) * K.log(preds) + (1 - labels)*K.pow(preds, gamma)*K.log(1 - preds))
        return K.mean(fl)
    

    def create_model(cell_type, if_bidirectional=False):
        if cell_type == 'LSTM':
            rnn_layer = tf.keras.layers.LSTM
        elif cell_type == 'GRU':
            rnn_layer = tf.keras.layers.GRU
        elif cell_type == 'RNN':
            rnn_layer = tf.keras.layers.RNN

        inputs = tf.keras.Input(shape=(steps, len(profiles_use)))
        x = inputs

        if if_bidirectional == True:
            for lstm_layer in range(num_lstm_layers):
                x = tf.keras.layers.Bidirectional(rnn_layer(num_units, return_sequences=True))(x)
            x = tf.keras.layers.Bidirectional(rnn_layer(num_units))(x)
        else:
            for lstm_layer in range(num_lstm_layers):
                x = rnn_layer(num_units, return_sequences=True)(x)
            x = rnn_layer(num_units)(x)

        for fc_layer in range(num_fc_layers):
            x = tf.keras.layers.Dense(num_neurons, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=[focal], metrics=['binary_crossentropy'])
        return model

    model = create_model(cell_type, if_bidirectional)
    print("input shape is", x_train.shape)
    history = model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[Metrics(x_train, y_train, train_id, dir_name)], verbose=1)
    return model

if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.load(f)

    with h5py.File("../data/lstm_norm_data/long_horizon/train/train_human_detectable.h5", 'r') as g:
        x_train = g["input"][...]
        y_train = g["output"][...]
        event_train = g["event)num"][...]

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
    
    dir_name = "./training_dynamics_all_human_detectable/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    training(x_train=x_train,
            y_train=y_train,
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
                                                    
