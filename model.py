# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:13:36 2017

@author: wolf4461
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    Merge,
    merge,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling2D,
    Flatten
    )
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import (
    Convolution2D)
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

weight_decay = 1e-4
def _conv_bn_relu(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        conv_b = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(act_a)
        norm_b = BatchNormalization()(conv_b)
        act_b = Activation(activation = 'relu')(norm_b)
        return act_b
    return f

def net_base(input, nb_filter = 64):
    # Stream
    block1 = _conv_bn_relu(nb_filter,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu(nb_filter,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu(nb_filter,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu(nb_filter,3,3)(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(block4)
    # =========================================================================
    block5 = _conv_bn_relu(nb_filter,3,3)(pool4)
    up5 = merge([UpSampling2D(size=(2, 2))(block5), block4], mode='concat', concat_axis=-1)
    # =========================================================================
    block6 = _conv_bn_relu(nb_filter,3,3)(up5)
    up6 = merge([UpSampling2D(size=(2, 2))(block6), block3], mode='concat', concat_axis=-1)
    # =========================================================================
    block7 = _conv_bn_relu(nb_filter,3,3)(up6)
    up7 = merge([UpSampling2D(size=(2, 2))(block7), block2], mode='concat', concat_axis=-1)
    # =========================================================================
    block8 = _conv_bn_relu(nb_filter,3,3)(up7)
    up8 = merge([UpSampling2D(size=(2, 2))(block8), block1], mode='concat', concat_axis=-1)
    # =========================================================================
    block9 = _conv_bn_relu(nb_filter,3,3)(up8)
    return block9

def buildModel (input_dim):
    # This network is used to pre-train the optical flow.
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = net_base (input_, nb_filter = 64 )
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, activation='linear',init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model