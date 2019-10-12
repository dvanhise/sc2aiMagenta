import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Concatenate, Reshape


class SpicyModel(tf.keras.Model):
    def __init__(self, dim_map, dim_depth, dim_acts, dim_args):
        super().__init__()

        # Inputs
        # screen_input = tf.keras.Input(shape=(dim_map, dim_map, dim_depth))
        # select_input = tf.keras.Input(shape=(7, 10))
        # available_actions = tf.keras.Input(shape=(dim_acts,))

        #  What is N?  Who's to know?
        N = 32

        self.conv1 = Conv2D(2*dim_depth, 3, padding='same', activation='relu',
                            input_shape=(dim_depth, dim_map, dim_map))
        self.conv2 = Conv2D(N, 3, padding='valid', activation='relu')
        self.maxpool = MaxPooling2D()
        self.screen_reshape = Reshape((15, 32))

        self.selected_dense = Dense(N, use_bias=True, activation='relu')
        self.selected_reshape = Reshape((7, 32))

        self.avail_act_dense = Dense(N, use_bias=True, activation='relu')
        self.avail_act_reshape = Reshape((1, 32))

        self.concat = Concatenate(axis=1)
        self.combined_dense = Dense(N, use_bias=True, activation='relu')
        self.combined_flatten = Flatten()

        self.args_dense = Dense(dim_args, use_bias=True, activation='sigmoid')
        self.act_probs_dense = Dense(dim_acts, use_bias=True, activation='softmax')

    def call(self, screen, selected, available_actions):
        # screen -> conv+maxpool -> conv+maxpool -> concat -> dense -> dense -> arguments
        # selected -> dense -----------------------^ ^            \-> dense -> action probs
        # available_actions -> dense ---------------/

        screen_conv = self.conv1(screen)
        screen_conv = self.maxpool(screen_conv)
        screen_conv = self.conv2(screen_conv)
        screen_conv = self.maxpool(screen_conv)
        screen_conv = self.screen_reshape(screen_conv)

        selected_part = self.selected_dense(selected)
        selected_part = self.selected_reshape(selected_part)

        avail_act_part = self.avail_act_dense(available_actions)
        avail_act_part = self.avail_act_reshape(avail_act_part)

        concat = self.concat([screen_conv, selected_part, avail_act_part])
        concat = self.combined_dense(concat)
        split = self.combined_flatten(concat)

        arguments = self.args_dense(split)
        action_probs = self.act_probs_dense(split)

        return action_probs, arguments
