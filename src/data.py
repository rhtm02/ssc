import useful_module
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

FEATURES = ['APS', 'CANRx_LONG_ACCEL', 'FlywheelTq', 'Ne', 'NiLocal', 'NoLocal', 'Slope',
            'TQI_J', 'cam_ClutchActuatorPos', 'ccm_ClutchActTorque_Nm', 'ccm_ClutchTgtPosition',
            'ccm_ClutchTgtTorque_Nm', 'csm_ClutchActState', 'gdm_CurGear', 'gdm_TgtGear',
            'gsm_DrivingMode_CAN', 'iom_Acc_real','iom_Longitudinal_Distance',
            'iom_Relative_Velocity' ,'iom_LvrPosition','iom_SlopeG_LongAccelFil','iom_SlopeG_VehSusPitch',
            'iom_VSP16', 'ssm_ClutchTgtGear', 'ssm_ClutchTgtState','BrakePress',
            'CANRx_CYL_PRES', 'cam_CltMotCur', 'cam_CltPedalPos_pct','CANRx_SAS_Angle', 'iom_ECU_Lat_Accel',
            'APS_BINARY', 'BrakeSwitchLocal_BINARY']

FEATURES_1 = ['APS', 'CANRx_LONG_ACCEL', 'FlywheelTq', 'Ne', 'NiLocal', 'NoLocal', 'Slope',
            'TQI_J', 'cam_ClutchActuatorPos', 'ccm_ClutchActTorque_Nm', 'ccm_ClutchTgtPosition',
            'ccm_ClutchTgtTorque_Nm', 'csm_ClutchActState', 'gdm_CurGear', 'gdm_TgtGear',
            'gsm_DrivingMode_CAN', 'iom_Acc_real','iom_Longitudinal_Distance',
            'iom_Relative_Velocity' ,'iom_LvrPosition','iom_SlopeG_LongAccelFil','iom_SlopeG_VehSusPitch',
            'iom_VSP16', 'ssm_ClutchTgtGear', 'ssm_ClutchTgtState']
FEATURES_2 = ['BrakePress', 'CANRx_CYL_PRES',
              'cam_CltMotCur', 'cam_CltPedalPos_pct']
FEATURES_3 = ['CANRx_SAS_Angle', 'iom_ECU_Lat_Accel']
FEATURES_ = ['APS', 'CANRx_LONG_ACCEL', 'FlywheelTq', 'Ne', 'NiLocal', 'NoLocal', 'Slope',
            'TQI_J', 'cam_ClutchActuatorPos', 'ccm_ClutchActTorque_Nm', 'ccm_ClutchTgtPosition',
            'ccm_ClutchTgtTorque_Nm', 'csm_ClutchActState', 'gdm_CurGear', 'gdm_TgtGear',
            'gsm_DrivingMode_CAN', 'iom_Acc_real','iom_Longitudinal_Distance',
            'iom_Relative_Velocity' ,'iom_LvrPosition','iom_SlopeG_LongAccelFil','iom_SlopeG_VehSusPitch',
            'iom_VSP16', 'ssm_ClutchTgtGear', 'ssm_ClutchTgtState','BrakePress',
            'CANRx_CYL_PRES', 'cam_CltMotCur', 'cam_CltPedalPos_pct','CANRx_SAS_Angle', 'iom_ECU_Lat_Accel']

def single_dataload(path = '../data/mean/validation/200610_PDe_E441489_Test_출근_광주2서울_mean.csv'
                          , y_label = 'BrakeSwitchLocal_BINARY', window_size = 50, predict = 12):
    data = pd.read_csv(path)[FEATURES]
    x_1 = data[FEATURES_1]
    x_2 = data[FEATURES_2]
    x_3 = data[FEATURES_3]
    y = data[y_label]
    x_1,_ = useful_module.make_3d_sequencial_data(window_size, x_1.values,y)
    x_2,_ = useful_module.make_3d_sequencial_data(window_size, x_2.values,y)
    x_3,y = useful_module.make_3d_sequencial_data(window_size, x_3.values,y)
    x_list, y = useful_module.make_ssc_dataset(predict_index=predict, x=[x_1,x_2,x_3], y=y)

    x_1 = x_list[0]
    x_2 = x_list[1]
    x_3 = x_list[2]
    y = y
    print(x_1.shape)
    print(x_2.shape)
    print(x_3.shape)
    print(y.shape)

    return x_1,x_2,x_3, y

def seq2seq_dataload(path, predict = 12):

    data = pd.read_csv(path)[FEATURES_]
    X,Y = useful_module.make_3d_seq2seq_data(predict, data.values)
    encorder_X = X[:-1]
    decorder_X = Y[:-1]
    Y = Y[1:]
    return encorder_X,decorder_X,Y

def seq2seq_val_dataload(path, predict = 12):

    data = pd.read_csv(path)[FEATURES_]
    X,Y = useful_module.make_3d_seq2seq_data(predict, data.values)
    encorder_X = X[:-1]
    decorder_X = []
    for i in encorder_X:
        decorder_X.append(i[-1])
    decorder_X = np.asarray(decorder_X)
    decorder_X = decorder_X.reshape((-1,1,31))
    Y = Y[1:]
    return encorder_X,decorder_X,Y
