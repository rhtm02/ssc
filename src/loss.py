import tensorflow as tf

class custom_loss:
    def __init__(self, Ti_ON_in = 30, Ti_ON_out = 15, Ti_OFF_in = 30, Ti_OFF_out = 15):
        self.Ti_ON_in = Ti_ON_in
        self.Ti_ON_out = Ti_ON_out
        self.Ti_OFF_in = Ti_OFF_in
        self.Ti_OFF_out = Ti_OFF_out
    def ssc_loss(self, y_true, y_pred):
        y_pred_ = y_pred
        y_true_ = tf.keras.backend.cast_to_floatx(y_true)
        #print(y_pred_, y_true_)
        error = tf.keras.backend.sum(y_true_ - y_pred_)
        error_square = tf.square(error)
        check = (error < 1)
        loss = tf.where(check, error_square/2, error_square)
        return loss
'''
loss = custom_loss()
print(loss.ssc_loss(tf.constant([0,0,0,0,1,1,1,1,1,0,0,0,0,0]), tf.constant([0.6,0.6,0.6,0.6,0.3,0.3,0.3,0.3,0.3,0.6,0.6,0.6,0.6,0.6], dtype=tf.float32)))
print(loss.ssc_loss(tf.constant([0,0,0,0,1,1,1,1,1,0,0,0,0,0]), tf.constant([0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6], dtype=tf.float32)))
print(loss.ssc_loss(tf.constant([0,0,0,0,1,1,1,1,1,0,0,0,0,0]), tf.constant([0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3], dtype=tf.float32)))
print(loss.ssc_loss(tf.constant([0,0,0,0,1,1,1,1,1,0,0,0,0,0]), tf.constant([0,0,0,0,1,1,1,1,1,0,0,0,0,0], dtype=tf.float32)))
'''
