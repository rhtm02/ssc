import models
import data
import tensorflow as tf
import pandas as pd
import gpu_setting
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
if(__name__ == '__main__'):
    gpu_setting.GPU_GROWTH()
    WINSIZE = 50
    for label in ['BrakeSwitchLocal_BINARY']:
        for i in range(13, 15):
            class_weight = {0: 1.,
                            1: (i + 1) / 2,
                            2: i,
                            3: 1.}
            print(class_weight)
            MODEL = models.SEP_LSTM()
            #earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
            MODEL.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
            print(MODEL.summary())
            valX_1 = []
            valX_2 = []
            valX_3 = []
            valY = []
            for val in os.listdir('../data/mean/validation/'):
                temp_valX_1, temp_valX_2, temp_valX_3, temp_valY = data.single_dataload(path='../data/mean/validation/'+val, y_label=label,
                                                         window_size=WINSIZE, predict = 12)
                valX_1 += temp_valX_1.tolist()
                valX_2 += temp_valX_2.tolist()
                valX_3 += temp_valX_3.tolist()
                valY += temp_valY.tolist()
            valX_1 = np.asarray(valX_1)
            valX_2 = np.asarray(valX_2)
            valX_3 = np.asarray(valX_3)
            valY = np.asarray(valY)

            for BATCHSIZE in [128]:
                    import numpy as np
                    Version = str(BATCHSIZE) + '_BATCH'
                    import os
                    file_list = os.listdir('../data/mean/train')
                    for file in file_list:
                        #print(valX_1.shape, valX_3.shape, valX_2.shape, valY.shape)
                        X_1,X_2,X_3,Y = data.single_dataload(path='../data/mean/train/' + file, y_label=label,
                                                             window_size=WINSIZE, predict = 12)
                        #class weights
                        Y_integers = np.argmax(Y, axis = 1)

                        modelcheck = tf.keras.callbacks.ModelCheckpoint(
                                    filepath='../result/submodule_train/' + label + '/' + str(WINSIZE) + '_' + Version +'_'+file.split('.')[0] +'_'+ str(i) + '_class_weight' +'.h5',
                                    save_weights_only=False,
                                    monitor='val_loss', mode='min', save_best_only=True)

                        history = MODEL.fit([X_1,X_2,X_3], Y, batch_size=BATCHSIZE, callbacks=[modelcheck]
                                                   ,validation_data=([valX_1,valX_2,valX_3], valY), epochs=40, verbose=2, class_weight = class_weight
                                               , shuffle=False)
                        save = pd.DataFrame(data=history.history)
                        save.to_csv('../result/submodule_train/'+label+'/'+ Version +'_'+file.split('.')[0] +'_' + str(i) + '_class_weight' +'_history.csv')
                    del X_1,X_2,X_3,Y,save

            tf.keras.backend.clear_session()
