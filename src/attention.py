import tensorflow as tf


class SEQ2SEQ(tf.keras.Model):
    def __init__(self, features=31):
        super(SEQ2SEQ, self).__init__()
        self.encorder = tf.keras.layers.LSTM(features, return_state=True, name='encorder')
        self.decorder = tf.keras.layers.LSTM(features, return_state=True, return_sequences=True, name='decorder')

    def call(self, inputs, training = False):
        encorder_input = inputs[0]
        decorder_input = inputs[1]
        encorder_output, state_h, state_c = self.encorder(encorder_input)
        decorder_output, h, c = self.decorder(decorder_input, initial_state=[state_h,state_c])
        return decorder_output,h,c

