import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Concatenate, Reshape, Input

from tensorflow.keras.backend import placeholder


class SpicyModel(tf.keras.Model):
    def __init__(self, screen_width, screen_height, screen_depth, ns_input_length, action_size):
        super().__init__()

        # Inputs
        # screen_input = tf.keras.Input(shape=(dim_map, dim_map, dim_depth))
        # select_input = tf.keras.Input(shape=(7, 10))
        # available_actions = tf.keras.Input(shape=(dim_acts,))

        self.screen_input = Input(shape=(screen_width, screen_height, screen_depth))
        self.ns_input = Input(shape=(ns_input_length,))

        self.conv1 = Conv2D(screen_depth, 5, strides=1, padding='same', activation='relu', input_shape=self.screen_input.shape)
        self.conv2 = Conv2D(2*screen_depth, 3, strides=1, padding='same', activation='relu')
        # self.maxpool = MaxPooling2D()

        self.ns_dense = Dense(screen_width, use_bias=True, activation='relu')

        self.concat = Concatenate(axis=2)

        self.conv_out = Conv2D(1, 3, padding='same', activation='relu')

        self.state_dense = Dense(256, use_bias=True, activation='relu')
        self.ns_actions_out = Conv2D(action_size, 1, strides=1, padding='same')
        self.value_out = Dense(1, use_bias=True)


    def call(self, screen, ns_data, training=False):
        # screen -> conv+maxpool -> conv+maxpool -> concat -> dense -> dense -> arguments
        # available_actions -> dense ---------------/             \-> dense -> action probs

        screen_conv = self.conv1(screen)
        # screen_conv = self.maxpool(screen_conv)
        screen_conv = self.conv2(screen_conv)
        # screen_conv = self.maxpool(screen_conv)

        ns = self.ns_dense(ns_data)

        state = self.concat([screen_conv, ns])
        state_ns = self.state_dense(state)

        value = self.value_out(state_ns)
        ns_action_policy = self.ns_actions_out(state_ns)

        spacial_action_policy = self.conv_out(state)

        return spacial_action_policy, ns_action_policy, value
