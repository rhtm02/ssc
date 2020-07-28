import pandas as pd
import tensorflow as tf
import data

model = tf.keras.models.load_model('../model/8_32_BPS.h5')

test_x, test_y = data.test_dataload(path = '../data/bps_aps_binary/mean/200610_PDe_E441489_Test_출근_광주2서울_mean.csv', y_label = 'BrakeSwitchLocal', window_size = 8)

y_true = pd.DataFrame(data = test_y, columns = ['y_true'])
y_pred = model.predict(test_x)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
y_pred = pd.DataFrame(data = y_pred, columns=['y_pred'])
result = pd.concat([y_true,y_pred], axis = 1)

result.to_csv('../result_bps.csv')
