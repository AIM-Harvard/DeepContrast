
import os
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications import (
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet50V2,
    ResNet101V2,
    ResNet152V2
    )



def ResNet(resnet, input_shape, transfer=False, freeze_layer=None, activation='sigmoid'):

    """
    ResNet: 50, 101, 152

    Args:
        resnet {str} -- resnets with different layers, i.e. 'ResNet101';
        input_shape {np.array} -- input data shape;
    
    Keyword args:
        transfer {boolean} -- decide if transfer learning;
        freeze_layer {int} -- number of layers to freeze;
        activation {str or function} -- activation function in last layer: 'sigmoid', 'softmax';
    
    Returns:
        ResNet model;
    
    """

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
    if resnet == 'ResNet50V2':
        base_model = ResNet50V2(
            weights=None,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None      
            )                
    elif resnet == 'ResNet101V2':
        base_model = ResNet101V2(
            weights=None,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None       
            )                
    elif resnet == 'ResNet152V2': 
        base_model = ResNet152V2(
            weights=None,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None       
            )               
    elif resnet == 'ResNet50':
        base_model = ResNet50(
            weights=None,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    elif resnet == 'ResNet101':
        base_model = ResNet101(
            weights=None,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    elif resnet == 'ResNet152':
        base_model = ResNet152(
            weights=None,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )

    ### create top model
    inputs = base_model.input
    x = base_model.output  
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
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





    

    
