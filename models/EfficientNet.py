
import os
import numpy as np
import pandas as pd
import glob
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.applications import ( 
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5
    )


def EfficientNet(effnet, input_shape, transfer=False, freeze_layer=None, activation='sigmoid'):


    """
    EfficientNets: B3, B4, B5;

    Args:
        effnet {str} -- EfficientNets with different layers;
        input_shape {np.array} -- input data shape;
    
    Keyword args:
        transfer {boolean} -- decide if transfer learning;
        freeze_layer {int} -- number of layers to freeze;
        activation {str or function} -- activation function in last layer: 'sigmoid', 'softmax';
    
    Returns:
        EfficientNet model;    
    
    """


    # determine if use transfer learnong or not
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

    ### determine ResNet base model
    if effnet == 'EffNetB3':
        base_model = EfficientNetB3(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    elif effnet == 'EffNetB4':
        base_model = EfficientNetB4(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    elif effnet == 'EffNetB5':
        base_model = EfficientNetB5(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    base_model.trainable = True

 ### create top model
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation=activation)(x)
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
    

    
