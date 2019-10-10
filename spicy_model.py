import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Concatenate


class SpicyModel(tf.keras.Model):
    def __init__(self, dim_map, dim_depth, dim_acts):
        super().__init__()

        # Inputs
        # screen_input = tf.keras.Input(shape=(dim_map, dim_map, dim_depth))
        # select_input = tf.keras.Input(shape=(7, 10))
        # available_actions = tf.keras.Input(shape=(dim_acts,))

        self.conv1 = Conv2D(2*dim_depth, 3, padding='same', activation='relu', input_shape=(dim_map, dim_map, dim_depth))
        self.conv2 = Conv2D(2*dim_depth, 3, padding='same', activation='relu')
        self.maxpool = MaxPooling2D()

        self.selected_dense = Dense(50)
        self.avail_act_dense = Dense(50)

        self.concat = Concatenate()
        self.combined_dense = Dense(50)

        self.args_dense = Dense(4, activation='sigmoid')
        self.act_probs_dense = Dense(dim_acts, activation='sigmoid')

    def call(self, screen, selected, available_actions, training=False):
        # screen -> conv+maxpool -> conv+maxpool -> concat -> full -> full -> arguments
        # selected -> full ------------------------^ ^            \-> full -> action probs
        # available_actions -> full ----------------/

        screen_conv = self.conv1(screen)
        screen_conv = self.maxpool(screen_conv)
        screen_conv = self.conv2(screen_conv)
        screen_conv = self.maxpool(screen_conv)

        selected_part = self.selected_dense(selected)
        avail_act_part = self.avail_act_dense(available_actions)

        concat = self.concat(screen_conv, selected_part, avail_act_part)
        split = self.combined_dense(concat)

        arguments = self.args_dense(split)
        action_probs = self.act_probs_dense(split)

        return action_probs, arguments
