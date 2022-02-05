#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2

# ----------------------------------------------------------------------------------
# transfer learning CNN model
# ----------------------------------------------------------------------------------
def GoogleNet(resnet, transfer_learning, input_shape, batch_momentum, activation,
              activation_out, loss_function, optimizer, dropout_rate):
    
    ### determine if use transfer learnong or not
    if transfer_learning == True:
        weights = 'imagenet'
    elif transfer_learning == False:
	    weights = None
    
	### dermine input shape
    default_shape = (224, 224, 3)
    if input_shape == default_shape:
	    include_top = True
    else:
	    include_top = False
    
    ### determine ResNet base model
    if resnet == 'ResNet50V2':
        base_model = ResNet50V2(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None      
            )                
    elif resnet == 'ResNet101V2':
        base_model = ResNet101V2(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None       
            )                
    elif resnet == 'ResNet152V2': 
        base_model = ResNet152V2(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None       
            )                
#    base_model.trainable = True
#	
#    inputs = keras.Input(shape=input_shape)
#    x = inputs
#    x = base_model(x, training=True)	
#    x = BatchNormalization(momentum=batch_momentum)(x)
#    x = GlobalAveragePooling2D()(x)
#    x = Dropout(0.2)(x)
#    outputs = Dense(1)(x)
#    model = Model(inputs, outputs) 
#    model.summary()
    ### create top model
    out = base_model.output  
    out = BatchNormalization(momentum=batch_momentum)(out)
    out = GlobalAveragePooling2D()(out)
    out = Dropout(dropout_rate)(out)
    ### layer 3
#    out = BatchNormalization(momentum=batch_momentum)(out)
#    out = Dense(512, activation=activation)(out)
#    out = Dropout(dropout_rate)(out)
#    ### lyaer 2
#    out = BatchNormalization(momentum=batch_momentum)(out)
#    out = Dense(128, activation=activation)(out)
#    out = Dropout(dropout_rate)(out)
#    ### layer 1
#    out = BatchNormalization(momentum=batch_momentum)(out)
    predictions = Dense(1, activation=activation_out)(out)
    model = Model(inputs=base_model.input, outputs=predictions)
    
	### only if we want to freeze layers
#    for layer in base_model.layers:
#        layer.trainable = True

    print('complie model')
    model.compile(
                  loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy']
                  )
    model.summary()

    return model





    

    
