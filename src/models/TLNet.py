
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import tensorflow
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2




def TLNet(resnet, input_shape, activation='sigmoid'):

    """
    Transfer learning based on ResNet

    Args:
        resnet {str} -- resnets with different layers, i.e. 'ResNet101';
        input_shape {np.array} -- input data shape;

    Keyword args:
        activation {str or function} -- activation function in last layer: 'sigmoid', 'softmax';

    Returns:
        Transfer learning model;


    """

    ## determine ResNet base model
    if resnet == 'ResNet50V2':
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None      
            )                
    elif resnet == 'ResNet101V2':
        base_model = ResNet101V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None       
            )                
    elif resnet == 'ResNet152V2': 
        base_model = ResNet152V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None       
            )               
    
    base_model.trainable = False
    
    ### create top model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
    #x = Dense(1024, activation='relu')(x)
    #x = Dense(512, activation='relu')(x)
    outputs = Dense(1, activation=activation)(x)
    model = Model(inputs, outputs)
  
    return model





    

    
