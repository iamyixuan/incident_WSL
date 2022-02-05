import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import viz_prediction

# load predictin data
# y_train = np.loadtxt('../data/saved_predictions/2020_jan_feb_mar_cleaned/y_train.txt')
# y_train_pred = np.loadtxt('../data/saved_predictions/2020_jan_feb_mar_cleaned/y_train_pred.txt')
y_test = np.loadtxt('../data/saved_predictions/y_test_2019_May_Jun.txt')
y_test_pred = np.loadtxt('../data/saved_predictions/y_test_pred_2019_May_Jun.txt')


viz_prediction(y_test_pred, y_test, 360, '../notes/plots/test_May_Jun_pred/')