import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
FEATURES = ['TQI_J', 'APS', 'gdm_TgtGear', 'CANRx_LONG_ACCEL',
       'iom_SlopeG_VehSusPitch', 'NSCs_Brk_ExTimeCnt', 'iom_LvrPosition',
       'cam_CltPedalPos_pct', 'HSSo_stNeutralCoasting', 'iom_SlopeG_APSDot',
       'Slope', 'cam_CltMotCur', 'ssm_ClutchTgtGear', 'ssm_ClutchTgtState',
       'CANRx_SAS_Angle', 'gsm_RoadGrade', 'NiLocal', 'BrakePress',
       'cam_ClutchActuatorPos', 'csm_ClutchActState', 'ccm_ClutchTgtTorque_Nm',
       'Ne', 'gdm_CurGear', 'ccm_ClutchActTorque_Nm', 'ccm_ClutchTgtPosition',
       'NSCs_Target', 'NSCs_PedalPos_ExTimeCnt', 'iom_SlopeG_LongAccelFil',
       'NSCs_TCU_SSC_Inhibit', 'clm_EngineIdleTargetMod_rpm', 'CANRx_CYL_PRES',
       'NSCs_EMS_SSC_Tgt', 'FlywheelTq', 'NoLocal', 'iom_VSP16',
       'ssm_EngineState', 'iom_Longitudinal_Distance', 'iom_Relative_Velocity',
       'iom_Acc_real', 'BrakeSwitchLocal', 'iom_ECU_Lat_Accel',
       'gsm_DrivingMode_CAN', 'NSCs_TCU_SSC_st', 'HSSo_boNeutralCoasting']
CONT = ['TQI_J', 'APS',  'CANRx_LONG_ACCEL',
       'iom_SlopeG_VehSusPitch', 'NSCs_Brk_ExTimeCnt', 
       'cam_CltPedalPos_pct', 'HSSo_stNeutralCoasting', 'iom_SlopeG_APSDot',
       'Slope', 'cam_CltMotCur',  
       'CANRx_SAS_Angle',  'NiLocal', 'BrakePress',
       'cam_ClutchActuatorPos',  'ccm_ClutchTgtTorque_Nm',
       'Ne', 'ccm_ClutchActTorque_Nm', 'ccm_ClutchTgtPosition',
        'NSCs_PedalPos_ExTimeCnt', 'iom_SlopeG_LongAccelFil',
        'clm_EngineIdleTargetMod_rpm', 'CANRx_CYL_PRES',
        'FlywheelTq', 'NoLocal', 'iom_VSP16',
       'iom_Longitudinal_Distance', 'iom_Relative_Velocity',
       'iom_Acc_real',  'iom_ECU_Lat_Accel']
DISC = ['gdm_TgtGear','iom_LvrPosition', 'ssm_ClutchTgtGear',
        'ssm_ClutchTgtState', 'gsm_RoadGrade', 'csm_ClutchActState',
        'gdm_CurGear', 'NSCs_Target', 'NSCs_TCU_SSC_Inhibit', 'NSCs_EMS_SSC_Tgt',
        'ssm_EngineState', 'BrakeSwitchLocal', 'gsm_DrivingMode_CAN', 
        'NSCs_TCU_SSC_st', 'HSSo_boNeutralCoasting']
MMS = MinMaxScaler()

def prepro(file, HZ = 10):
    print(file)
    CANdata_minmax = pd.read_csv("../data/hyundai/minmax.csv")[CONT]
    CANdata = pd.read_csv("../data/hyundai/주행데이터0601_0610/" + file)
    #CANdata = CANdata.astype(float)
    CANdata['BrakeSwitchLocal'] = CANdata['BrakeSwitchLocal'].apply(lambda x : 0 if (x <= 1) else x)
    CANdata['BrakeSwitchLocal'] = CANdata['BrakeSwitchLocal'].apply(lambda x : 1 if (x >= 2) else x)
    
    CANdata['APS'] = CANdata['APS'].apply(lambda x : 0 if (x == 0) else x)
    CANdata['APS'] = CANdata['APS'].apply(lambda x : 1 if (x > 0) else x)
    file = file.split('.')[0]
    
    #처리할 데이터
    preCAN_new = CANdata
    preCAN_new = preCAN_new.dropna()
    preCAN_new = preCAN_new.reset_index(drop=True)
    preCAN_new_cont = preCAN_new[CONT]
    preCAN_new_disc = preCAN_new[DISC]
    
    #10번째 마다 추출
    preCAN_new_10hz = preCAN_new[(preCAN_new.index%int(100/HZ) == 0)]
    preCAN_new_10hz = preCAN_new_10hz.reset_index(drop=True)
    preCAN_new_10hz_cont = preCAN_new_10hz[CONT]
    preCAN_new_10hz_disc = preCAN_new_10hz[DISC]
    preCAN_new_10hz_cont = preCAN_new_10hz_cont.append(CANdata_minmax)
    preCAN_new_10hz_cont_mms = pd.DataFrame(data = MMS.fit_transform(preCAN_new_10hz_cont),
                                        columns=CONT).iloc[:-2]
    SAVE_data = pd.concat([preCAN_new_10hz_cont_mms, preCAN_new_10hz_disc], axis = 1)
    SAVE_data.to_csv("../data/hyundai/pick/" + file + '_pick.csv')
    print(SAVE_data.shape)
    #평균 추출
    CAN_cont_10hz = []
    CAN_disc_10hz = []
    for i in range(1,len((preCAN_new))):
        if (i%10 == 0):
            CAN_cont_mean = np.asarray(preCAN_new_cont[(preCAN_new_cont.index > (i-int(100/HZ))) & 
                                          (preCAN_new_cont.index <= i)].mean())
            CAN_disc_mode = np.asarray(preCAN_new_disc[(preCAN_new_cont.index > (i-int(100/HZ))) & 
                                          (preCAN_new_cont.index <= i)].mode())
            #print(CAN_disc_mode, CAN_disc_mode.shape)
            if(CAN_disc_mode.shape[0] == 2):
                #print(CAN_disc_mode, CAN_disc_mode.shape)
                #print(preCAN_new_disc[(preCAN_new_cont.index > (i-int(100/HZ))) & 
                #                          (preCAN_new_cont.index <= i)].values)
                for index in range(len(DISC)):
                    if (not np.isnan(CAN_disc_mode[1][index])):
                        CAN_disc_mode[0][index] = preCAN_new_disc[(preCAN_new_cont.index > (i-int(100/HZ))) & 
                                          (preCAN_new_cont.index <= i)].values[-1][index]
                #print(CAN_disc_mode[0], CAN_disc_mode.shape)
                #print(" ")
                CAN_disc_10hz.append(CAN_disc_mode[0])
            else:
                CAN_disc_10hz.append(CAN_disc_mode[0])
            CAN_cont_10hz.append(CAN_cont_mean)
    CAN_cont_10hz.append(CANdata_minmax.values[0])
    CAN_cont_10hz.append(CANdata_minmax.values[1])
    CAN_disc_10hz = pd.DataFrame(data = CAN_disc_10hz, columns=DISC)
    CAN_cont_10hz_mms = pd.DataFrame(data = MMS.fit_transform(CAN_cont_10hz), columns=CONT)[:-2]
    SAVE_data = pd.concat([CAN_cont_10hz_mms,CAN_disc_10hz], axis = 1)
    SAVE_data.to_csv("../data/hyundai/mean/" + file + '_mean.csv')
    print(SAVE_data.shape)

import os
file_list = os.listdir("../data/hyundai/주행데이터0601_0610/")
for i in file_list:
    if ('변수' in i):
        continue
    elif ('minmax' in i):
        continue
    prepro(file = i)





