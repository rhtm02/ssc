import tensorflow as tf
import pandas as pd
import data
import models

FEATURES = ['TQI_J', 'gdm_TgtGear', 'CANRx_LONG_ACCEL',
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
       'iom_Acc_real', 'iom_ECU_Lat_Accel',
       'gsm_DrivingMode_CAN']
for feature in FEATURES:
    for label in ['APS','BrakeSwitchLocal']:
            EPOCH = 1000
            BATCHSIZE = 500
            Version = feature
            X, Y, valX, valY = data.dataload_select(path='../data/bps_aps_binary/mean', train_ratio=0.8, x_features = feature, y_label=label, window_size=32)
            LSTM = models.BDLSTM(X, path=Version, unit=10, label=label)

            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
            LSTM.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
            print(LSTM.summary())

            history = LSTM.fit(X, Y, batch_size = BATCHSIZE ,callbacks=[modelcheck, earlystopping], validation_data=(valX, valY), epochs=EPOCH,
                                   verbose=1)
            save = pd.DataFrame(data=history.history)
            save.to_csv('../result/' + label + '/' + Version + '/history.csv')
            tf.keras.backend.clear_session()

