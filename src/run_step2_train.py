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
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from go_model.data_generator import train_generator
from go_model.data_generator import val_generator
from go_model.get_model import get_model
from go_model.train_model import train_model
from go_model.train_model import callbacks
from opts import parse_opts



if __name__ == '__main__':

    opt = parse_opts()

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    if opt.root_dir is not None:
        opt.HN_out_dir = os.path.join(opt.root_dir, opt.HN_out)
        opt.HN_model_dir = os.path.join(opt.root_dir, opt.HN_model)
        opt.HN_label_dir = os.path.join(opt.root_dir, opt.HN_label)
        opt.HN_pro_data_dir = os.path.join(opt.root_dir, opt.HN_pro_data)
        opt.HN_log_dir = os.path.join(opt.root_dir, opt.HN_log)
        if not os.path.exists(opt.HN_out_dir):
            os.makedirs(opt.HN_out_dir)
        if not os.path.exists(opt.HN_model_dir):
            os.makedirs(opt.HN_model_dir)
        if not os.path.exists(opt.HN_label_dir):
            os.makedirs(opt.HN_label_dir)
        if not os.path.exists(opt.HN_pro_data_dir):
            os.makefirs(opt.HN_pro_data_dir)
        if not os.path.exists(opt.HN_log_dir):
            os.makedirs(opt.HN_log_dir)

    print('\n--- STEP 2 - TRAIN MODEL ---\n')

    # data generator for train and val data
    train_gen = train_generator(
        pro_data_dir=opt.HN_pro_data_dir,
        batch_size=opt.batch_size)
    x_val, y_val, val_gen = val_generator(
        pro_data_dir=opt.HN_pro_data_dir,
        batch_size=opt.batch_size)

    # get CNN model 
    my_model = get_model(
        out_dir=opt.HN_out_dir,
        run_model=opt.run_model, 
        activation=opt.activation, 
        input_shape=opt.input_shape,
        freeze_layer=opt.freeze_layer, 
        transfer=opt.transfer)

    ### train model
    if not opt.no_train:
        if opt.optimizer_function == 'adam':
            optimizer = Adam(learning_rate=opt.lr)
        train_model(
            root_dir=opt.root_dir,
            out_dir=opt.HN_out_dir,
            log_dir=opt.HN_log_dir,
            model_dir=opt.HN_model_dir,
            model=my_model,
            run_model=opt.run_model,
            train_gen=train_gen,
            val_gen=val_gen,
            x_val=x_val,
            y_val=y_val,
            batch_size=opt.batch_size,
            epoch=opt.epoch,
            optimizer=optimizer,
            loss_function=opt.loss_function,
            lr=opt.lr)

