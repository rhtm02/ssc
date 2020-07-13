#for useful method code
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def draw_confusion(predict, ground_truth,label_num = 6, dir = '',score = 0):

    predict = np.argmax(predict,axis=-1)
    cm = confusion_matrix(ground_truth, predict)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[str(i) for i in range(label_num)]
                , yticklabels=[str(i) for i in range(label_num)], cmap='Blues', square=True)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(score)
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.savefig(dir + '_cm.png')
    plt.show()

def make_3d_sequencial_data(window_size, x):
    # x,y is numpy values
    input_len = x.shape[0]
    # output data shape
    X = np.zeros([input_len - (window_size - 1), window_size, x.shape[1]])
    for i in range(input_len - (window_size - 1)):
        # print(x[i:i+ window_size].reshape(1,window_size,x.shape[1]))
        X[i] = x[i:i + window_size].reshape(1, window_size, x.shape[1])
    return X
def make_2d_sequencial_data(window_size, x):
    # x,y is numpy values
    input_len = x.shape[0]
    # output data shape
    X = np.zeros([input_len - (window_size - 1), window_size,1])
    for i in range(input_len - (window_size - 1)):
        # print(x[i:i+ window_size].reshape(1,window_size,x.shape[1]))
        X[i] = x[i:i + window_size].reshape(1, window_size,1)
    return X
def make_ssc_dataset(predict_index, x, y):
    Y = y.values[x.shape[1] + predict_index - 1:]
    X = x[:Y.shape[0]]
    return X, Y