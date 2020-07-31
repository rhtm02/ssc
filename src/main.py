import models
import data
import tensorflow as tf
import pandas as pd

from loss import custom_loss

if(__name__ == '__main__'):
    loss = custom_loss()

    for label in ['APS','BrakeSwitchLocal']:
        EPOCH = 1000
        BATCHSIZE = 500
        if(label == 'APS'):
            FEATURE_LEN = len(data.APS_FEATURES)
        else:
            FEATURE_LEN = len(data.BPS_FEATURES)
        for UNIT in [1,2,4,8,16,32,64, 128]:
            Version = str(UNIT)

            LSTM = models.BDLSTM(path = Version, unit = UNIT, label = label, shape = (None, FEATURE_LEN))
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
            LSTM.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
            print(LSTM.summary())
            save = pd.DataFrame()
            for WINSIZE in [2, 4, 8, 16, 32, 64, 128, 256]:
                X, Y, valX, valY = data.dataload(path='../data/bps_aps_binary/mean', train_ratio=0.8, y_label=label,
                                                 window_size=WINSIZE)
                print(X.shape)
                modelcheck = tf.keras.callbacks.ModelCheckpoint(
                    filepath='../result/' + label + '/' + Version + '/' +str(WINSIZE) + '_' +  Version + '.h5', save_weights_only=False,
                    monitor='val_loss', mode='min', save_best_only=True)

                history = LSTM.fit(X,Y,batch_size = BATCHSIZE,callbacks=[modelcheck,earlystopping] ,validation_data = (valX,valY),epochs = EPOCH, verbose = 1)
                temp = pd.DataFrame(data = history.history)
                save.append(temp)
                del X,Y,valX,valY
            save.to_csv('../result/' + label + '/' + Version + '/history.csv')
            tf.keras.backend.clear_session()