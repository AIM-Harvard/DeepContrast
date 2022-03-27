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
import random
from tensorflow.keras.optimizers import Adam
from go_model.evaluate_model import evaluate_model
from statistics.write_txt import write_txt
from statistics.get_stats_plots import get_stats_plots
from opts import parse_opts
import tensorflow as tf



if __name__ == '__main__':

    opt = parse_opts()

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    if opt.root_dir is not None:
        opt.HN_out_dir = os.path.join(opt.root_dir, opt.HN_out)
        opt.HN_model_dir = os.path.join(opt.root_dir, opt.HN_model)
        opt.HN_pro_data_dir = os.path.join(opt.root_dir, opt.HN_pro_data)
        if not os.path.exists(opt.HN_out_dir):
            os.makedirs(opt.HN_out_dir)
        if not os.path.exists(opt.HN_model_dir):
            os.makedirs(opt.HN_model_dir)
        if not os.path.exists(opt.HN_pro_data_dir):
            os.makefirs(opt.HN_pro_data_dir)
     
    print('\n--- STEP 3 - MODEL EVALUATION ---\n')   
    
    for opt.run_type in ['val', 'test']:
        # evalute model
        loss, acc = evaluate_model(
            run_type=opt.run_type,
            model_dir=opt.HN_model_dir,
            pro_data_dir=opt.HN_pro_data_dir,
            saved_model=opt.run_model,
            threshold=opt.thr_img,
            activation=opt.activation)
        if opt.stats_plots:
            get_stats_plots(
                pro_data_dir=opt.HN_pro_data_dir,
                root_dir=opt.root_dir,
                run_type=opt.run_type,
                run_model=opt.run_model,
                loss=loss,
                acc=acc,
                saved_model=opt.run_model,
                epoch=opt.epoch,
                batch_size=opt.batch_size,
                lr=opt.lr,
                thr_img=opt.thr_img,
                thr_prob=opt.thr_prob,
                thr_pos=opt.thr_pos,
                bootstrap=opt.n_bootstrap)
