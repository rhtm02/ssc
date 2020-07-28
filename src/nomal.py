import pandas as pd
import data

test_x, test_y = data.test_dataload(path = '../data/bps_aps_binary/mean/200610_PDe_E441489_Test_출근_광주2서울_mean.csv', y_label = 'APS', window_size = 1)

y_true = pd.read_csv('../data/bps_aps_binary/mean/200610_PDe_E441489_Test_출근_광주2서울_mean.csv')['APS'].iloc[:-15]

y_true = pd.DataFrame(data = y_true.values, columns = ['y_true'])
y_pred = pd.DataFrame(data = test_y, columns=['y_stride'])
result = pd.concat([y_true,y_pred], axis = 1)

result.to_csv('../normal_aps.csv')
test_x, test_y = data.test_dataload(path = '../data/bps_aps_binary/mean/200610_PDe_E441489_Test_출근_광주2서울_mean.csv', y_label = 'BrakeSwitchLocal', window_size = 1)

y_true = pd.read_csv('../data/bps_aps_binary/mean/200610_PDe_E441489_Test_출근_광주2서울_mean.csv')['BrakeSwitchLocal'].iloc[:-15]

y_true = pd.DataFrame(data = y_true.values, columns = ['y_true'])
y_pred = pd.DataFrame(data = test_y, columns=['y_stride'])
result = pd.concat([y_true,y_pred], axis = 1)

result.to_csv('../normal_bps.csv')