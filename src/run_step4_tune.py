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
import random
import argparse
from tensorflow.keras.optimizers import Adam
from go_model.tune_model import tune_model
from statistics.write_txt import write_txt
from statistics.get_stats_plots import get_stats_plots
from go_model.evaluate_model import evaluate_model
from train_data.tune_dataset import tune_pat_dataset
from train_data.tune_dataset import tune_img_dataset
from opts import parse_opts
import tensorflow as tf

  

if __name__ == '__main__':

    opt = parse_opts()

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    if opt.root_dir is not None:
        opt.CH_out_dir = os.path.join(opt.root_dir, opt.CH_out)
        opt.CH_data_dir = os.path.join(opt.root_dir, opt.CH_data)
        opt.CH_label_dir = os.path.join(opt.root_dir, opt.CH_label)
        opt.CH_pro_data_dir = os.path.join(opt.root_dir, opt.CH_pro_data)
        opt.CH_pre_data_dir = os.path.join(opt.root_dir, opt.CH_pre_data)
        opt.CH_model_dir = os.path.join(opt.root_dir, opt.CH_model)
        opt.HN_model_dir = os.path.join(opt.root_dir, opt.HN_model)
        if not os.path.exists(opt.CH_out_dir):
            os.makedirs(opt.CH_out_dir)
        if not os.path.exists(opt.CH_data_dir):
            os.makedirs(opt.CH_data_dir)
        if not os.path.exists(opt.CH_label_dir):
            os.makedirs(opt.CH_label_dir)
        if not os.path.exists(opt.CH_pro_data_dir):
            os.makefirs(opt.CH_pro_data_dir)
        if not os.path.exists(opt.CH_pre_data_dir):
            os.makedirs(opt.CH_pre_data_dir)
        if not os.path.exists(opt.CH_model_dir):
            os.makedirs(opt.CH_model_dir)
    
    print('\n--- STEP 4 - MODEL FINE TUNE ---\n')   
    
    # get chest CT data anddo  preprocessing
    if opt.get_CH_data:
        tune_pat_dataset(
            data_dir=opt.CH_data_dir,
            pre_data_dir=opt.CH_pre_data_dir,
            pro_data_dir=opt.CH_pro_data_dir,
            label_dir=opt.CH_label_dir,
            label_file=opt.CH_label_file,
            crop_shape=opt.CH_crop_shape)
        tune_img_dataset(
            pro_data_dir=opt.CH_pro_data_dir,
            pre_data_dir=opt.CH_pre_data_dir,
            slice_range=opt.CH_slice_range)

    ## fine tune models by freezing some layers
    if opt.fine_tune:
        opt.run_type = 'tune'
        tuned_model, model_fn = tune_model(
            HN_model_dir=opt.HN_model_dir,
            CH_model_dir=opt.CH_model_dir, 
            pro_data_dir=opt.CH_pro_data_dir, 
            HN_model=opt.run_model,
            batch_size=opt.batch_size, 
            epoch=opt.epoch, 
            freeze_layer=opt.freeze_layer)
    
    ## evaluate finetuned model on chest CT data
    if opt.test:
        opt.run_type = 'tune'
        loss, acc = evaluate_model(
            run_type=opt.run_type,
            model_dir=opt.CH_model_dir,
            pro_data_dir=opt.CH_pro_data_dir,
            saved_model=opt.tuned_model)
        if opt.stats_plots:
            get_stats_plots(
                pro_data_dir=opt.CH_pro_data_dir,
                root_dir=opt.root_dir,
                run_type=opt.run_type,
                run_model=opt.run_model,
                loss=0,
                acc=0,
                saved_model=opt.tuned_model,
                epoch=opt.epoch,
                batch_size=opt.batch_size,
                lr=opt.lr)


    
