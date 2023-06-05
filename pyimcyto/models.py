import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import backend as keras
from tensorflow.keras import Input

def nested_unet(pretrained_weights = None,input_size = (256,256,1), loss_function = 'binary_crossentropy'):
    inputs = Input(shape=input_size)

    # leftmost diagnoal of nested unet:
    conv_0_0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv_0_0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_0_0)
    pool_0_0 = MaxPooling2D(pool_size=(2, 2))(conv_0_0)

    conv_1_0 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool_0_0)
    conv_1_0 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1_0)
    pool_1_0 = MaxPooling2D(pool_size=(2, 2))(conv_1_0)

    conv_2_0 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool_1_0)
    conv_2_0 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_2_0)
    pool_2_0 = MaxPooling2D(pool_size=(2, 2))(conv_2_0)

    conv_3_0 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool_2_0)
    conv_3_0 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_3_0)
    drop_3_0 = Dropout(0.5)(conv_3_0)
    pool_3_0 = MaxPooling2D(pool_size=(2, 2))(drop_3_0)

    conv_4_0 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool_3_0)
    conv_4_0 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_4_0)
    drop_4_0 = Dropout(0.5)(conv_4_0)

    # second diagonal of nested unet:
    up_0_1 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_1_0))
    merge_0_1 = concatenate([conv_0_0, up_0_1], axis = 3)
    conv_0_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_0_1)
    conv_0_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_0_1)

    up_1_1 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_2_0))
    merge_1_1 = concatenate([conv_1_0, up_1_1], axis = 3)
    conv_1_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_1_1)
    conv_1_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1_1)

    up_2_1 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_3_0))
    merge_2_1 = concatenate([conv_2_0, up_2_1], axis = 3)
    conv_2_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_2_1)
    conv_2_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_2_1)

    up_3_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop_4_0))
    merge_3_1 = concatenate([drop_3_0, up_3_1], axis = 3)
    conv_3_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_3_1)
    conv_3_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_3_1)

    # third diagonal of nested unet:
    up_0_2 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_1_1))
    merge_0_2 = concatenate([conv_0_0, conv_0_1, up_0_2], axis = 3)
    conv_0_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_0_2)
    conv_0_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_0_2)
    
    up_1_2 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_2_1))
    merge_1_2 = concatenate([conv_1_0, conv_1_1, up_1_2], axis = 3)
    conv_1_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_1_2)
    conv_1_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1_2)

    up_2_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_3_1))
    merge_2_2 = concatenate([conv_2_0, conv_2_1, up_2_2], axis = 3)
    conv_2_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_2_2)
    conv_2_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_2_2)
    
    # fourth diagonal of nested unet:
    up_0_3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_1_2))
    merge_0_3 = concatenate([conv_0_0, conv_0_1, conv_0_2, up_0_3], axis = 3)
    conv_0_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_0_3)
    conv_0_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_0_3)

    up_1_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_2_2))
    merge_1_3 = concatenate([conv_1_0, conv_1_1, conv_1_2, up_1_3], axis = 3)
    conv_1_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_1_3)
    conv_1_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1_3)

    # fifth diagonal of nested unet:
    up_0_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_1_3))
    merge_0_4 = concatenate([conv_0_0, conv_0_1, conv_0_2, conv_0_3, up_0_4], axis = 3)
    conv_0_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_0_4)
    conv_0_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_0_4)
    conv_0_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_0_4)
    conv_sig = Conv2D(1, 1, activation = 'sigmoid')(conv_0_4)

    model = Model(inputs = inputs, outputs = conv_sig)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = loss_function, metrics = ['accuracy'])
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def simple_autoencoder(input_features, pretrained_weights = None, learning_rate=1e-4):
    
    drop = 0.1

    model = Sequential()

    model.add(Dense(input_features, input_dim=input_features, activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(input_features//2, activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(input_features//4, activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(input_features//2, activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(input_features, activation='softmax'))

    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer = Adam(lr = learning_rate), metrics=['accuracy'])
    model.compile(loss='MSLE', 
                optimizer = 'adam', 
                metrics=['MSE'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    # model.summary()

    return model

def error_model(features, model_weights):

    n_features = len(features)
    model = simple_autoencoder(n_features, model_weights)
    return model