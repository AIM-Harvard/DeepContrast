
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def train_generator(pro_data_dir, batch_size, input_channel=3):

    """
    create data generator for training dataset;

    Arguments:
        out_dir {path} -- path to output results;
        batch_size {int} -- batch size for data generator;
        input_channel {int} -- input channel for image;

    Return:
        Keras data generator;
    
    """
    
    ### load train data based on input channels
    if input_channel == 1:
        fn = 'train_arr_1ch.npy'
    elif input_channel == 3:
        #fn = 'train_arr_3ch_crop.npy'
        fn = 'train_arr_3ch.npy'
    x_train = np.load(os.path.join(pro_data_dir, fn))

    ### load val labels
    train_df = pd.read_csv(os.path.join(pro_data_dir, 'train_img_df.csv'))
    y_train  = np.asarray(train_df['label']).astype('int').reshape((-1, 1))

    ## data generator
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype=None,
        )

   ### Train generator
    train_gen = datagen.flow(
        x=x_train,
        y=y_train,
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        )
    print('Train generator created')

    return train_gen


def val_generator(pro_data_dir, batch_size, input_channel=3):

    """
    create data generator for validation dataset;

    Arguments:
        out_dir {path} -- path to output results;
        batch_size {int} -- batch size for data generator;
        input_channel {int} -- input channel for image;

    Return:
        Keras data generator;

    """
    
    ### load val data based on input channels
    if input_channel == 1:
        fn = 'val_arr_1ch.npy'
    elif input_channel == 3:
        fn = 'val_arr_3ch.npy'
    x_val = np.load(os.path.join(pro_data_dir, fn))

    ### load val labels
    val_df = pd.read_csv(os.path.join(pro_data_dir, 'val_img_df.csv'))
    y_val = np.asarray(val_df['label']).astype('int').reshape((-1, 1))

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype=None,
        )
    
    datagen = ImageDataGenerator()
    val_gen = datagen.flow(
        x=x_val,
        y=y_val,
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        )
    print('val generator created')

    return x_val, y_val, val_gen



