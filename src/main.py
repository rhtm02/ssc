import models
import data
import tensorflow as tf
import pandas as pd

from loss import custom_loss

if(__name__ == '__main__'):
    loss = custom_loss()

    for label in ['BrakeSwitchLocal']:
        for WINSIZE in [32,64]:
            EPOCH = 1000
            BATCHSIZE = 200
            for UNIT in [1,2,4,8,16,32,64]:
                Version = str(WINSIZE) + '_' + str(UNIT)
                X,Y,valX,valY = data.dataload(path = '../data/bps_aps_binary/mean',train_ratio = 0.8, y_label = label, window_size = WINSIZE)
                LSTM = models.BDLSTM(X, path = Version, unit = UNIT, label = label)
                modelcheck = tf.keras.callbacks.ModelCheckpoint(filepath='../result/' + label + '/' + Version + '/' + Version + '.h5', save_weights_only=False, monitor='val_loss',mode='min',save_best_only=True)
                earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
                LSTM.compile(optimizer = 'rmsprop', loss = loss.ssc_loss, metrics = ['accuracy'])
                print(LSTM(X[1:2])[0][0].numpy())
                print(LSTM.summary())
                history = LSTM.fit(X,Y,callbacks=[modelcheck,earlystopping] ,validation_data = (valX,valY),epochs = EPOCH, verbose = 1)
                save = pd.DataFrame(data = history.history)
                save.to_csv('../result/' + label + '/' + Version + '/history.csv')
                tf.keras.backend.clear_session()