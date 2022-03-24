import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
from train_data.get_img_dataset import get_img_dataset
from train_data.get_pat_dataset import get_pat_dataset
from train_data.preprocess_data import preprocess_data




if __name__ == '__main__':

    proj_dir = 'root path'
    data_dir = os.path.join(proj_dir, 'HeadNeck/raw_image')
    pre_data_dir = os.path.join(proj_dir, 'HeadNeck/pre_image')
    label_dir = os.path.join(proj_dir, 'HeadNeck/label')
    out_dir = os.path.join(proj_dir, 'HeadNeck/output')
    pro_data_dir = os.path.join(proj_dir, 'HeadNeck/pro_data')
    label_file = 'HN_label.csv'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(pre_data_dir):
        os.makedirs(pre_data_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(pro_data_dir):
        os.makedirs(pro_data_dir)

    print('\n--- STEP 1 - GET DATA ---\n')

    preprocess_data(
        data_dir=data_dir,
        pre_data_dir=pre_data_dir,
        new_spacing=(1, 1, 3),
        data_exclude=None,
        crop_shape=[192, 192, 100]
        )

    data_tot, label_tot, ID_tot = get_pat_dataset(
        data_dir=data_dir,
        pre_data_dir=pre_data_dir,
        label_dir=label_dir,
        label_file=label_file,
        pro_data_dir=pro_data_dir,
        )

    get_img_dataset(
        pro_data_dir=pro_data_dir,
        run_type=None,
        data_tot=data_tot,
        ID_tot=ID_tot,
        label_tot=label_tot,
        slice_range=range(17, 83),
        )



