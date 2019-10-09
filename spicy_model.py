import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


class SpicyModel(tf.keras.Model):
    def __init__(self, dim_map, dim_depth, dim_output):
        super().__init__()

        # (22, 22, 10) -> (15,)

        self.model = Sequential([
            Conv2D(2*dim_depth, 3, padding='same', activation='relu', input_shape=(dim_map, dim_map, dim_depth)),
            MaxPooling2D(),
            Conv2D(2*dim_depth, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(2*dim_depth, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(dim_output**2, activation='relu'),
            Dense(dim_output, activation='sigmoid')
            # Split off action policy tensor and run softmax on it
        ])

    def call(self, inputs, training=False):
        pass

    # with tf.Session() as sess:
    #   save_path = saver.save(sess, "/tmp/model.ckpt")
    #   saver.restore(sess, "/tmp/model.ckpt")
