import useful_module
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
FEATURES = ['TQI_J', 'APS', 'gdm_TgtGear', 'CANRx_LONG_ACCEL',
       'iom_SlopeG_VehSusPitch', 'iom_LvrPosition',
       'cam_CltPedalPos_pct', 'iom_SlopeG_APSDot',
       'Slope', 'cam_CltMotCur', 'ssm_ClutchTgtGear', 'ssm_ClutchTgtState',
       'CANRx_SAS_Angle', 'gsm_RoadGrade', 'NiLocal', 'BrakePress',
       'cam_ClutchActuatorPos', 'csm_ClutchActState', 'ccm_ClutchTgtTorque_Nm',
       'Ne', 'gdm_CurGear', 'ccm_ClutchActTorque_Nm', 'ccm_ClutchTgtPosition',
        'iom_SlopeG_LongAccelFil',
        'clm_EngineIdleTargetMod_rpm', 'CANRx_CYL_PRES',
        'FlywheelTq', 'NoLocal', 'iom_VSP16',
       'ssm_EngineState', 'iom_Longitudinal_Distance', 'iom_Relative_Velocity',
       'iom_Acc_real', 'BrakeSwitchLocal', 'iom_ECU_Lat_Accel',
       'gsm_DrivingMode_CAN']
#70% 이상
APS_FEATURES = ['TQI_J', 'gdm_TgtGear', 'CANRx_LONG_ACCEL', 'iom_LvrPosition',
                'ssm_ClutchTgtGear', 'ssm_ClutchTgtState', 'NiLocal',
                'csm_ClutchActState', 'Ne', 'gdm_CurGear', 'FlywheelTq',
                'iom_VSP16', 'gsm_DrivingMode_CAN']
#80% 이상
BPS_FEATURES = ['TQI_J', 'gdm_TgtGear', 'iom_LvrPosition', 'ssm_ClutchTgtGear',
                'BrakePress', 'Ne', 'gdm_CurGear', 'CANRx_CYL_PRES', 'FlywheelTq',
                'NoLocal', 'iom_VSP16', 'gsm_DrivingMode_CAN']

def test_dataload(path = '../data/bps_aps_binary/mean/200610_PDe_E441489_Test_출근_광주2서울_mean.csv'
                          , y_label = 'BrakeSwitchLocal', window_size = 50):
    data = pd.read_csv(path)[FEATURES]
    if(y_label == 'BrakeSwitchLocal'):
        x = data[BPS_FEATURES]
    else:
        x = data[APS_FEATURES]
    y = data[y_label]
    x = useful_module.make_3d_sequencial_data(window_size, x.values)
    test_x,test_y = useful_module.make_ssc_dataset(predict_index = 15,x = x,y = y)
    return test_x, test_y

def dataload(path = '../data/bps_aps_binary/mean',train_ratio = 0.8
                          , y_label = 'BrakeSwitchLocal', window_size = 50):
    file_list = os.listdir(path)
    file_list = sorted(file_list)
    #print(file_list)
    X = []
    Y = []

    for file in file_list[:-1]:
        #print(file)
        data = pd.read_csv(path + '/' + file)[FEATURES]
        if (y_label == 'BrakeSwitchLocal'):
            x = data[BPS_FEATURES]
        else:
            x = data[APS_FEATURES]
        y = data[y_label]
        x = useful_module.make_3d_sequencial_data(window_size, x.values)
        x,y = useful_module.make_ssc_dataset(predict_index = 15,x = x,y = y)
        X += x.tolist()
        Y += y.tolist()
    X = np.asarray(X)
    Y = np.asarray(Y)
    train_x = X[:int(len(X)*train_ratio)]
    train_y = Y[:int(len(X)*train_ratio)]
    val_x = X[int(len(X)*train_ratio):]
    val_y = Y[int(len(X)*train_ratio):]
    #print(train_x.shape, (train_y==0).sum(),(train_y==1).sum(), val_x.shape, (val_y==0).sum(), (val_y==1).sum())

    X = np.asarray(X)
    Y = np.asarray(Y)
    train_x = X[:int(len(X) * train_ratio)]
    train_y = Y[:int(len(X) * train_ratio)]
    val_x = X[int(len(X) * train_ratio):]
    val_y = Y[int(len(X) * train_ratio):]
    return train_x, train_y, val_x, val_y

def dataload_select(path = '../data/bps_aps_binary/mean',train_ratio = 0.8, x_features = 'TQI_J'
                          , y_label = 'BrakeSwitchLocal', window_size = 50):
    file_list = os.listdir(path)
    file_list = sorted(file_list)
    #print(file_list)
    X = []
    Y = []
    for file in file_list[:-1]:
        #print(file)
        data = pd.read_csv(path + '/' + file)[FEATURES]
        x = data[x_features]
        y = data[y_label]
        x = useful_module.make_2d_sequencial_data(window_size, x.values)
        x,y = useful_module.make_ssc_dataset(predict_index = 15,x = x,y = y)
        X += x.tolist()
        Y += y.tolist()
    X = np.asarray(X)
    Y = np.asarray(Y)
    train_x = X[:int(len(X)*train_ratio)]
    train_y = Y[:int(len(X)*train_ratio)]
    val_x = X[int(len(X)*train_ratio):]
    val_y = Y[int(len(X)*train_ratio):]
    print(train_x.shape, (train_y==0).sum(),(train_y==1).sum(), val_x.shape, (val_y==0).sum(), (val_y==1).sum())
    #print(X.tolist())
    return train_x, train_y, val_x, val_y

#dataload()