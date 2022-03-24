import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pltimport glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
from tensorflow.keras.optimizers import Adam
import argparse
from get_data.get_img_dataset import get_img_dataset
from get_data.get_pat_dataset import get_pat_dataset
from get_data.preprocess_data import preprocess_data
from opts import parse_opts



if __name__ == '__main__':
    
    opt = parse_opts()
    if opt.root_dir != '':
        opt.data_dir = os.path.join(opt.root_dir, opt.data_dir)
        opt.pre_data_dir = os.path.join(opt.root_dir, opt.pre_data_dir)
        opt.label_dir = os.path.join(opt.root_dir, opt.label_dir)
        opt.out_dir = os.path.join(opt.root_dir, opt.out_dir)
        opt.pro_data_dir = os.path.join(opt.root_dir, opt.pro_data_dir)
        opt.model_dir = os.path.join(opt.root_dir, opt.model_dir)
        opt.log_dir = os.path.join(opt.root_dir, opt.log_dir)
        opt.train_dir = os.path.join(opt.root_dir, opt.train_dir)
        opt.val_dir = os.path.join(opt.root_dir, opt.val_dir)
        opt.test_dir = os.path.join(opt.root_dir, opt.test_dir)
        if not os.path.exists(pre_data_dir):
            os.path.mkdir(pre_data_dir)
        if not os.path.exists(out_dir):
            os.path.mkdir(out_dir)
        if not os.path.exists(model_dir):
            os.path.mkdir(model_dir)
        if not os.path.exists(log_dir):
            os.path.mkdir(log_dir)
        if not os.path.exists(train_dir):
            os.path.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.path.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.path.mkdir(test_dir)

    dataset(
        data_dir=opt.data_dir, 
        pre_data_dir=opt.pre_data_dir, 
        pro_data_dir=opt.pro_data_dir, 
        label_dir=opt.label_dir, 
        data_exclude=opt.data_exclude, 
        new_spacing=opt.new_spacing, 
        crop_shape=opt.crop_shape,
        interp_type=opt.interp_type, 
        run_type=opt.run_type, 
        slice_range=opt.slice_range)
    
    if not opt.no_train:
        # data generator
        train_gen = train_generator(
            pro_data_dir=opt.pro_data_dir,
            batch_size=opt.batch_size)
        x_val, y_val, val_gen = val_generator(
            pro_data_dir=opt.pro_data_dir,
            batch_size=opt.batch_size)
        # get model
        my_model = get_model(
            out_dir=opt.out_dir,
            run_model=opt.run_model,
            activation=opt.activation,
            input_shape=opt.input_shape)
        ### train model
        if opt.optimizer_function == 'adam':
            optimizer = Adam(learning_rate=opt.lr)
        train(
            log_dir=opt.log_dir,
            model_dir=opt.model_dir,
            model=my_model,
            run_model=opt.run_model,
            train_gen=train_gen,
            val_gen=val_gen,
            x_val=x_val,
            y_val=y_val,
            batch_size=opt.batch_size,
            epoch=opt.epoch,
            optimizer=optimizer_function,
            loss_function=opt.loss_function,
            lr=opt.lr)

    if not opt.no_test:
        for run_type in ['val', 'test']:
            loss, acc = evaluate_model(
                run_type=run_type,
                model_dir=model_dir,
                pro_data_dir=pro_data_dir,
                saved_model=saved_model,
                threshold=0.5,
                activation='sigmoid')
            # get statistic and plots
            get_stats_plots(
                out_dir=out_dir,
                proj_dir=proj_dir,
                run_type=run_type,
                run_model=run_model,
                loss=None,
                acc=None,
                saved_model=saved_model,
                epoch=epoch,
                batch_size=batch_size,
                lr=lr,
                thr_img=0.5,
                thr_prob=0.5,
                thr_pos=0.5,
                bootstrap=1000)
