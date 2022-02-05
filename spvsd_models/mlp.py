from os import X_OK
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import MinMax, scores, get_detection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class FCnet(nn.Module):
    def __init__(self, input_shape, out_shape1, out_shape2):
        super(FCnet, self).__init__()
        self.fc1 = nn.Linear(input_shape, out_shape1)
        self.fc2 = nn.Linear(out_shape1, out_shape1)
        self.fc3 = nn.Linear(out_shape1, out_shape2)
        self.fc4 = nn.Linear(out_shape2, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = F.sigmoid(self.fc4(x))
        return output 

class OneDConvNet(nn.Module):
    def __init__(self, filter_size=3, num_channels=7, seq_length=20):
        super(OneDConvNet, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 2 * num_channels, filter_size, stride=2)
        self.shape_out_conv1 = int((seq_length - (filter_size - 1) - 1) / 2 + 1)
        self.conv2 = nn.Conv1d(2 * num_channels, 10, filter_size, stride=2)
        self.shape_out_conv2 = int((self.shape_out_conv1 - (filter_size - 1) - 1) / 2 + 1)
        self.fc1 = nn.Linear(self.shape_out_conv2 * 10, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.shape_out_conv2 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.sigmoid(self.fc3(x))
        return output

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim , 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        tag_space = self.linear(lstm_out[:, -1, :].view(x.shape[0], -1))
        out = F.relu(tag_space)
        out = F.relu(self.linear2(out))
        out = F.sigmoid(self.linear3(out))
        return out

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = torch.tensor([3]) #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def train(network, 
         x_train, 
         y_train, 
         x_val, 
         y_val,
         scaler, 
         profile_to_use, 
         horizon, 
         num_epochs, 
         batch_size, 
         learning_rate):

    num_iterations = x_train.shape[0] // batch_size
    steps = int(horizon * 60 / 30)
    shuffle_train = np.random.permutation(x_train.shape[0])
    print(shuffle_train.shape)

    x_val_org = x_val

    x_train = x_train[:, profile_to_use, -steps:]
    x_val = x_val[:, profile_to_use, -steps:]

    x_train = x_train[shuffle_train]

    if scaler!=None:
        x_train = scaler.fit_transform(x_train.reshape(-1, len(profile_to_use))).reshape(x_train.shape)
        x_val = scaler.transform(x_val.reshape(-1, len(profile_to_use))).reshape(x_val.shape)

    
    best_score = np.inf # depends on the metric if lower the better or higher the better
    train_ls = []
    val_ls = []
    if network == "FCNN":
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_val = np.reshape(x_val, (x_val.shape[0], -1))
        net = FCnet(x_train.shape[1], 200, 100)
    elif network == "1dCNN":
        net = OneDConvNet(num_channels=len(profile_to_use), seq_length=x_train.shape[2])
    elif network == "LSTM":
        net = LSTM(len(profile_to_use), 3, 1)
        x_train = np.transpose(x_train, [0, 2, 1])
        x_val = np.transpose(x_val, [0, 2, 1])
    else:
        raise ValueError("Choose networks between FCNN or 1dCNN")
    
    criterion = FocalLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    x_val, y_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()

    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0
        for iter in range(num_iterations):
            net.zero_grad()
            output = net(x_train[iter * batch_size : (1 + iter) * batch_size])
            loss = criterion(output, y_train[iter * batch_size : (1 + iter) * batch_size])
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
        
        train_ls.append(running_loss / num_iterations)

        net.eval()
        val_out = net(x_val)
        val_loss = criterion(val_out, y_val)
        val_ls.append(float(val_loss.item()))

        print("Epoch %d; Training loss is %.3f; Validation loss is %.3f" %(epoch + 1, train_ls[-1], val_ls[-1]))

    plt.plot(train_ls, label="Traning loss")
    plt.plot(val_ls, label="Validation loss")
    plt.show()

    net.eval()
    y_pred_t = net(x_val)
    label_pred = np.where(y_pred_t > 0.5, 1, 0)

    y_pred_train = net(x_train)
    label_pred_train = np.where(y_pred_train > 0.5, 1, 0)

    print(classification_report(y_train, label_pred_train))
    print(classification_report(y_val, label_pred))

    # detections = get_detection(y_pred_t.detach().numpy(), x_val_org, event_num_val, 0, "lol")
    # detections = np.concatenate(detections).reshape(-1, 1)
    # print(scores(detections, y_val, event_num_val))
    return net   

if __name__ == "__main__":
    with h5py.File('../data/lstm_norm_data/long_horizon/selected_train_from_original.h5','r') as f:
        x_train = f['input'][...]
        y_train = f['output'][...].reshape(-1, 1)
        #event_num_train = f['event_num'][...]

    with h5py.File('../data/lstm_norm_data/long_horizon/selected_val_from_original.h5','r') as f:
        x_val = f['input'][...]
        y_val = f['output'][...].reshape(-1, 1)
        #event_num_val = f['event)num'][...]

    def add_diff(x):
        speed_diff = (x[:, 3, :] - x[:, 0, :]) / 0.5 * (x[:, 3, :] + x[:, 0, :])
        occ_diff = (x[:, 5, :] - x[:, 2, :]) / 0.5 * (x[:, 5, :] + x[:, 2, :])
        speed_diff = np.expand_dims(speed_diff, axis=1)
        occ_diff = np.expand_dims(occ_diff, axis=1)
        x = np.concatenate([x, speed_diff, occ_diff], axis=1)
        return x


    # x_train = add_diff(x_train)
    # x_val = add_diff(x_val)

    # with h5py.File('../data/lstm_norm_data/long_horizon/test/test.h5', 'r') as f:
    #     x_test = f['input'][...]
    #     y_test = f['output'][...]
    #     event_num_test = f['event_num'][...]

    net = train(network="LSTM",
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                scaler=None,
                profile_to_use=[0, 3, 7],  # 0: upspeed; 1: upvolume; 2: upocc
                horizon=10,
                num_epochs=150,
                batch_size=64,
                learning_rate=0.0005)
  
