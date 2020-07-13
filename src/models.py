import tensorflow as tf
import sklearn
import os

def LSTM(X,path = 'V1', unit = 10):
    save_path = '../result/' + path
    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)

    input = tf.keras.layers.Input(shape = (X.shape[1],X.shape[2]))
    X = tf.keras.layers.LSTM(units=unit)(input)
    X = tf.keras.layers.Dense(units = unit)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dense(1)(X)
    output = tf.keras.activations.relu(X)

    return tf.keras.models.Model(input, output)

