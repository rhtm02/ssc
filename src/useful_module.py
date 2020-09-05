#for useful method code
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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

def make_3d_sequencial_data(window_size, x,y):
    # x,y is numpy values
    input_len = x.shape[0]
    # output data shape
    X = np.zeros([input_len - (window_size - 1), window_size, x.shape[1]])
    for i in range(input_len - (window_size - 1)):
        # print(x[i:i+ window_size].reshape(1,window_size,x.shape[1]))
        X[i] = x[i:i + window_size].reshape(1, window_size, x.shape[1])
    #print(y[window_size - 1:])
    Y = make_label_4(np.asarray(y[window_size - 1:]))
    return X, Y

def make_3d_seq2seq_data(window_size, x):
    # x,y is numpy values
    input_len = x.shape[0]
    # output data shape
    X = np.zeros([input_len - (window_size - 1), window_size, x.shape[1]])
    for i in range(input_len - (window_size - 1)):
        # print(x[i:i+ window_size].reshape(1,window_size,x.shape[1]))
        X[i] = x[i:i + window_size].reshape(1, window_size, x.shape[1])
    #print(y[window_size - 1:])
    Y = np.asarray(X[window_size - 1:])
    X = np.asarray(X[:-(window_size - 1)])
    return X, Y

def make_ssc_dataset(predict_index, x, y):
    #X is input list
    Y = y[predict_index:]
    X = []
    for i in x:
        X.append(i[:Y.shape[0]])
        print(i[:Y.shape[0]].shape, Y.shape)
    return X, np.asarray(Y)

def make_label_4(y):
    # x,y is numpy array
    OOE = OneHotEncoder()
    Y = [0]
    for index in range(1,len(y)):
        if((y[index - 1] == 0) and (y[index] == 0)):
            Y.append(0)
        elif((y[index - 1] == 1) and (y[index] == 1)):
            Y.append(1)
        elif((y[index - 1] == 0) and (y[index] == 1)):
            Y.append(2)
        else:
            Y.append(3)
    Y = np.asarray(Y)
    Y = Y.reshape(-1,1)
    OOE.fit([[0],[1],[2],[3]])
    Y = OOE.transform(Y).toarray()
    return np.asarray(Y)
def make_label_2(y):
    # x,y is numpy array
    OOE = OneHotEncoder()

    Y = y.reshape(-1,1)
    OOE.fit([[0],[1]])
    Y = OOE.transform(Y).toarray()
    return np.asarray(Y)

def label4_inverse(y):
    Y = []
    for temp in y[1:]:
        i = np.argmax(temp)
        print(temp,i)
        if(len(Y) == 0):
            if(i == 0):
                Y.append(0)
                Y.append(0)
            elif(i == 1):
                Y.append(1)
                Y.append(1)
            elif(i == 2):
                Y.append(0)
                Y.append(1)
            elif (i == 3):
                Y.append(1)
                Y.append(0)
        else:
            if(i == 0):
                Y.append(0)
            elif(i == 1):
                Y.append(1)
            elif(i == 2):
                Y.append(1)
            elif (i == 3):
                Y.append(0)
    Y.insert(0,0)
    Y = np.asarray(Y)
    return Y
'''
x = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24]]
y = [0,1,0,0,1,0]
X = pd.DataFrame(data = x, columns=['1','2','3','4'])
Y = pd.DataFrame(data = y, columns=['1'])
X,Y = make_3d_sequencial_data(2,X.values,Y.values)
print(Y)
Y_ = label4_inverse(Y)
for i in range(len(y)):
    print(y[i],Y_[i])
for i in range(len(X)):
    print(X[i],Y[i])
X,Y = make_ssc_dataset(2,[X],Y)
#print(Y)
for i in range(len(X[0])):
    print(X[0][i],Y[i])

x = np.asarray([[1,2,3,4],[2,6,7,8],[3,10,11,12],[4,14,15,16],[5,18,19,20],[6,22,23,24],[7,26,27,28],[8,31,32,33],[9,35,36,37]])
X,Y = make_3d_seq2seq_data(2,x)
a = []
print(X[:-1])
for i in X[:-1]:
    a.append(i[-1])
print(a)
print(Y[1:])
'''
