import attention
import training
import data
import tensorflow as tf
import gpu_setting
import os
gpu_setting.GPU_GROWTH()

EPOCH = 100

SEQ2SEQ = attention.SEQ2SEQ()
##### make dataset #####
train_files = os.listdir('../data/mean/train')
train_path = []
for file in train_files:
    train_path.append('../data/mean/train/' + file)
val_files = os.listdir('../data/mean/validation')
val_path = []
for file in val_files:
    val_path.append('../data/mean/validation/' + file)
VAL_EN_X_LIST = []
VAL_DE_X_LIST = []
VAL_Y_LIST = []
for file in val_path:
    VAL_EN_X, VAL_DE_X, VAL_Y = data.seq2seq_val_dataload(path=file,predict=12)
    VAL_EN_X_LIST.append(VAL_EN_X)
    VAL_DE_X_LIST.append(VAL_DE_X)
    VAL_Y_LIST.append(VAL_Y)
VAL_DATASET = tf.data.Dataset.from_tensor_slices((VAL_EN_X,VAL_DE_X,VAL_Y)).batch(128)

for train_data in train_path:
    EN_X,DE_X,Y = data.seq2seq_dataload(path=train_data,predict=12)
    TRAIN_DATA = (EN_X,DE_X,Y)
    TRAIN_DATASET =  tf.data.Dataset.from_tensor_slices(TRAIN_DATA).batch(128)
    SEQ2SEQ = training.training(model=SEQ2SEQ,train_dataset=TRAIN_DATASET,val_dataset=VAL_DATASET,epoch=EPOCH)