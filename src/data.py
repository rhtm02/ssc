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

def dataload(path = '../data/bps_aps_binary/mean',validation_ratio = 0.1
                          , y_label = 'BrakeSwitchLocal', window_size = 50):
    file_list = os.listdir(path)
    file_list = sorted(file_list)
    X = []
    Y = []
    for file in file_list:
        data = pd.read_csv(path + '/' + file)[FEATURES]
        x = data[FEATURES]
        y = data[y_label][window_size:]
        x = useful_module.make_3d_sequencial_data(window_size, x.values)
        x,y = useful_module.make_ssc_dataset(predict_index = 15,x = x,y = y)
        X += x.tolist()
        Y += y.tolist()
    X = np.asarray(X)
    Y = np.asarray(Y)
    train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=validation_ratio, shuffle=None, random_state=123)
    #print(train_x.shape, (train_y==0).sum(),(train_y==1).sum(), val_x.shape, (val_y==0).sum(), (val_y==1).sum())
    #print(X.tolist())
    return train_x, train_y, val_x, val_y