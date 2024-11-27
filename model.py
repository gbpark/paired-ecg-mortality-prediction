import tensorflow as tf
from tensorflow.keras.layers import (Input, GaussianNoise, Dense, Flatten, GlobalAveragePooling1D,
                                     BatchNormalization, Activation, Conv1D, MaxPooling1D, Multiply, Add)
from tensorflow.keras import Model
from tensorflow.keras import backend as K

def conv1d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = Conv1D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides, use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def SE_block(input_tensor, reduction_ratio=16):
    ch_input = K.int_shape(input_tensor)[-1]
    ch_reduced = ch_input // reduction_ratio
    x = GlobalAveragePooling1D()(input_tensor)
    x = Dense(ch_reduced, kernel_initializer='he_normal', activation='relu', use_bias=False)(x)
    x = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(x)
    x = tf.keras.layers.Reshape((1, ch_input))(x)
    x = Multiply()([input_tensor, x])
    return x

def SE_residual_block(input_tensor, filter_sizes, strides=1, reduction_ratio=16):
    filter_1, filter_2, filter_3 = filter_sizes
    x = conv1d_bn(input_tensor, filter_1, 1, strides=strides)
    x = conv1d_bn(x, filter_2, 3)
    x = conv1d_bn(x, filter_3, 1, activation=None)
    x = SE_block(x, reduction_ratio)
    projected_input = conv1d_bn(input_tensor, filter_3, 1, strides=strides, activation=None) if K.int_shape(input_tensor)[-1] != filter_3 else input_tensor
    shortcut = Add()([projected_input, x])
    return Activation('relu')(shortcut)

def stage_block(input_tensor, filter_sizes, blocks, reduction_ratio=16, stage=''):
    strides = 2 if stage != '2' else 1
    x = SE_residual_block(input_tensor, filter_sizes, strides, reduction_ratio)
    for _ in range(blocks - 1):
        x = SE_residual_block(x, filter_sizes, reduction_ratio=reduction_ratio)
    return x

def se_resnet(input_shape):
    inputs = Input(shape=input_shape)
    inputs = GaussianNoise(1e-3)(inputs)
    stage_1 = conv1d_bn(inputs, 64, 50, strides=10, padding='same')
    stage_1 = MaxPooling1D(3, strides=2, padding='same')(stage_1)
    stage_2 = stage_block(stage_1, [64, 64, 256], 3, stage='2')
    stage_3 = stage_block(stage_2, [128, 128, 512], 4, stage='3')
    stage_4 = stage_block(stage_3, [256, 256, 1024], 6, stage='4')
    stage_5 = stage_block(stage_4, [512, 512, 2048], 3, stage='5')
    output = GlobalAveragePooling1D()(stage_5)
    output = Flatten()(output)
    return Model(inputs, output)

class RandomChannelMask(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.mask = self.add_weight(shape=(input_shape[-1],), initializer='ones', trainable=False)
        self.mask = tf.tensor_scatter_nd_update(self.mask, [[0], [1]], [1.0, 1.0])

    def call(self, inputs, training=None):
        if training:
            random_values = tf.round(tf.random.uniform((self.mask.shape[0] - 2,), dtype=tf.float32))
            mask_on_the_fly = tf.tensor_scatter_nd_update(self.mask, [[2], [3], [4], [5], [6], [7]], random_values)
            return inputs * mask_on_the_fly
        return inputs

def get_compiled_model():
    input1 = Input(shape=(5000, 8), name='input1')
    input2 = Input(shape=(5000, 8), name='input2')
    resnet = se_resnet((5000, 8))
    input1 = RandomChannelMask()(input1)
    input2 = RandomChannelMask()(input2)
    layer1 = resnet(input1)
    layer2 = resnet(input2)
    layer = tf.keras.layers.Concatenate()([layer1, layer2])
    layer = Dense(1024, activation='relu', kernel_initializer='he_normal')(layer)
    output = Dense(1, activation='linear', kernel_initializer='he_normal')(layer)
    model = Model(inputs=[input1, input2], outputs=[output])
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, clipnorm=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model
