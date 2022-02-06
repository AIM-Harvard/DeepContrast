
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy




def simple_cnn(input_shape, activation):

    """
    simple CNN model
    
    Args:
        activation {str or function} -- activation function in last layer: 'softmax' or 'sigmoid';

    Returns:
        simple cnn model

    """
    
    ## determine n_output
    if activation == 'softmax':
        n_output = 2
    elif activation == 'sigmoid':
        n_output = 1

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.95))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(BatchNormalization(momentum=0.95))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))
    
    model.add(BatchNormalization(momentum=0.95))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(BatchNormalization(momentum=0.95))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(BatchNormalization(momentum=0.95))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization(momentum=0.95))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_output, activation=activation))

    model.summary()

    return model




    
