import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
import pydot
import pydotplus
import graphviz
#from pydotplus import graphviz
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from go_model.data_generator import train_generator
from go_model.data_generator import val_generator
from go_model.get_model import get_model
from go_model.train_model import train_model
from go_model.train_model import callbacks


if __name__ == '__main__':

    proj_dir = 'root path'
    label_dir = os.path.join(proj_dir, 'HeadNeck/label')
    out_dir = os.path.join(proj_dir, 'HeadNeck/output')
    pro_data_dir = os.path.join(proj_dir, 'HeadNeck/pro_data')
    log_dir = os.path.join(proj_dir, 'HeadNeck/log')
    model_dir = os.path.join(proj_dir, 'HeadNeck/model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)i
    
    batch_size = 32
    lr = 1e-5
    epoch = 1
    activation = 'sigmoid' #  'softmax' 
    loss_function = BinaryCrossentropy(from_logits=True)
    optimizer = Adam(learning_rate=lr)
    run_model = 'EffNetB4'    
    
    # data generator for train and val data
    train_gen = train_generator(
        pro_data_dir=pro_data_dir,
        batch_size=batch_size,
        )

    x_val, y_val, val_gen = val_generator(
        pro_data_dir=pro_data_dir,
        batch_size=batch_size,
        )

    my_model = get_model(
        out_dir=out_dir,
        run_model=run_model, 
        activation=activation, 
        input_shape=(192, 192, 3),
        freeze_layer=None, 
        transfer=False
        )

    ### train model
    train_model(
        out_dir=out_dir,
        log_dir=log_dir,
        model_dir=model_dir,
        model=my_model,
        run_model=run_model,
        train_gen=train_gen,
        val_gen=val_gen,
        x_val=x_val,
        y_val=y_val,
        batch_size=batch_size,
        epoch=epoch,
        optimizer=optimizer,
        loss_function=loss_function,
        lr=lr
        )

