import tensorflow as tf
import sklearn
import os

def LSTM(path = 'V1', unit = 10, label ='', shape = (None, 36)):
    save_path = '../result/' + label + '/' + path
    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)

    input = tf.keras.layers.Input(shape = shape)
    X = tf.keras.layers.LSTM(units=unit)(input)
    X = tf.keras.layers.Dense(units = 10)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dense(1)(X)
    output = tf.keras.activations.sigmoid(X)

    return tf.keras.models.Model(input, output)

def BDLSTM(path = 'V1', unit = 10, label = '', shape = (None, 36), stateful = False):
    save_path = '../result/' + label + '/' + path
    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)

    input = tf.keras.layers.Input(shape = (shape[1],shape[2]), batch_size=shape[0])
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=unit, stateful = stateful))(input)
    X = tf.keras.layers.Dense(units = 100)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(units=50)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(4)(X)
    output = tf.keras.activations.softmax(X)

    return tf.keras.models.Model(input, output)



def SEP_LSTM(path = 'submodule_train', shapes = [25,4,2]):
    #shapes = [clut,engine,bps,aps,trm,slp,lat,etc]
    save_path = '../result/' + path
    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)

    input_tensor_1 = tf.keras.layers.Input(shape = (None,shapes[0]))
    input_tensor_2 = tf.keras.layers.Input(shape = (None,shapes[1]))
    input_tensor_3 = tf.keras.layers.Input(shape = (None,shapes[2]))

    X_1 = tf.keras.layers.LSTM(units=shapes[0])(input_tensor_1)
    X_2 = tf.keras.layers.LSTM(units=shapes[1])(input_tensor_2)
    X_3 = tf.keras.layers.LSTM(units=shapes[2])(input_tensor_3)

    X = tf.concat([X_1,X_2,X_3],axis = 1)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=60)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(units=120)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(30)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(15)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(6)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(3)(X)
    X = tf.keras.activations.relu(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(4)(X)
    output = tf.keras.activations.softmax(X)

    return tf.keras.models.Model([input_tensor_1,input_tensor_2,input_tensor_3], output)



