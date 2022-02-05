#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as s1n
import matplotlib.pyplot as plt
import glob
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19


# ----------------------------------------------------------------------------------
# transfer learning CNN model
# ----------------------------------------------------------------------------------
def VGGNet(VGG, transfer, freeze_layer, input_shape, activation):

    ### Keras CNN models for use: https://keras.io/api/applications/
    ### InceptionV3(top1 acc 0.779)
    ### InceptionResnetV2(top1 acc 0.803),
    ### ResNet152V2(top1 acc 0.780)
    ### EficientNetB4

 ### determine if use transfer learnong or not
    if transfer == True:
        weights = 'imagenet'
    elif transfer == False:
        weights = None

    ### determine input shape
    default_shape = (224, 224, 3)
    if input_shape == default_shape:
        include_top = True
    else:
        include_top = False

    ## determine n_output
    if activation == 'softmax':
        n_output = 2
    elif activation == 'sigmoid':
        n_output = 1
    
    ### determine ResNet base model
    if VGG == 'VGG16':
        base_model = VGG16(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    elif VGG == 'VGG19':
        base_model = VGG19(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )

 ### create top model
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_output, activation=activation)(x)
    model = Model(inputs=inputs, outputs=outputs)

    ### freeze specific number of layers
    if freeze_layer == 1:
        for layer in base_model.layers[0:5]:
            layer.trainable = False
        for layer in base_model.layers:
            print(layer, layer.trainable)
    if freeze_layer == 5:
        for layer in base_model.layers[0:16]:
            layer.trainable = False
        for layer in base_model.layers:
            print(layer, layer.trainable)
    else:
        for layer in base_model.layers:
            layer.trainable = True
    model.summary()

    return model
    

    
