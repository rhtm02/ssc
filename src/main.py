import models
import data
import tensorflow as tf
import pandas as pd

EPOCH = 1000
BATCHSIZE = 200
UNIT = 5
WINSIZE = 50
Version = str(WINSIZE) + '_' + str(UNIT)

X,Y,valX,valY = data.dataload(path = '../data/bps_aps_binary/mean',validation_ratio = 0.2, y_label = 'BrakeSwitchLocal', window_size = WINSIZE)

LSTM = models.LSTM(X, path = Version, unit = UNIT)
modelcheck = tf.keras.callbacks.ModelCheckpoint(filepath='../result/' + Version + '/' + Version + '.h5', save_weights_only=False, monitor='val_loss',mode='min',save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

LSTM.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = LSTM.fit(X,Y,callbacks=[modelcheck,earlystopping] ,validation_data = (valX,valY),epochs = EPOCH, verbose = 1)
save = pd.DataFrame(data = history.history)
save.to_csv('../result/' + Version + '/history.csv')