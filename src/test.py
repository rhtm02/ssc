import pandas as pd
import tensorflow as tf
import data
import os
import numpy as np
file_list = os.listdir('../model/')
test_x_1,test_x_2,test_x_3, test_y= data.single_dataload(path='../data/mean/test/200612_PDe_E441489_Test_퇴근_mean.csv',
                                                                  y_label='APS_BINARY', window_size=50, predict = 12)
#print(test_y)
model = tf.keras.models.load_model('../model/APS.h5')
true_list = []
for i in test_y:
    true_list.append(np.argmax(i))
y_true = pd.DataFrame(data = true_list, columns = ['y_true'])
y_pred = model.predict([test_x_1,test_x_2,test_x_3])
pred_list = []
for i in y_pred:
    pred_list.append(np.argmax(i))
#print(true_list)
#print(pred_list)

y_pred = pd.DataFrame(data = pred_list, columns=['y_pred'])
result = pd.concat([y_true,y_pred], axis = 1)
result.to_csv('../APS_0612_퇴근.csv')
