import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from time import gmtime, strftime
from datetime import datetime
import timeit
import random
import argparse
from opts import parse_opts
import tensorflow as tf
from train_data.get_img_dataset import get_img_dataset
from train_data.get_pat_dataset import get_pat_dataset
from train_data.preprocess_data import preprocess_data



if __name__ == '__main__':
    
    opt = parse_opts()
    
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    if opt.root_dir is not None:
        opt.HN_out_dir = os.path.join(opt.root_dir, opt.HN_out)
        opt.HN_data_dir = os.path.join(opt.root_dir, opt.HN_data)
        opt.HN_label_dir = os.path.join(opt.root_dir, opt.HN_label)
        opt.HN_pro_data_dir = os.path.join(opt.root_dir, opt.HN_pro_data)
        opt.HN_pre_data_dir = os.path.join(opt.root_dir, opt.HN_pre_data)
        if not os.path.exists(opt.HN_out_dir):
            os.makedirs(opt.HN_out_dir)
        if not os.path.exists(opt.HN_data_dir):
            os.makedirs(opts.HN_data_dir)
        if not os.path.exists(opt.HN_label_dir):
            os.makedirs(opt.HN_label_dir)       
        if not os.path.exists(opt.HN_pro_data_dir):
            os.makefirs(opt.HN_pro_data_dir)
        if not os.path.exists(opt.HN_pre_data_dir):
            os.makedirs(opt.HN_pre_data_dir)
    
    print('\n--- STEP 1 - GET DATA ---\n')

    if opt.preprocess_data:
        preprocess_data(
            data_dir=opt.HN_data_dir,
            pre_data_dir=opt.HN_pre_data_dir,
            new_spacing=opt.new_spacing,
            data_exclude=opt.data_exclude,
            crop_shape=opt.HN_crop_shape)

    data_tot, label_tot, ID_tot = get_pat_dataset(
        data_dir=opt.HN_data_dir,
        pre_data_dir=opt.HN_pre_data_dir,
        label_dir=opt.HN_label_dir,
        label_file=opt.HN_label_file,
        pro_data_dir=opt.HN_pro_data_dir)

    get_img_dataset(
        pro_data_dir=opt.HN_pro_data_dir,
        run_type=None,
        data_tot=data_tot,
        ID_tot=ID_tot,
        label_tot=label_tot,
        slice_range=opt.HN_slice_range)



