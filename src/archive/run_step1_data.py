import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pltimport glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
from get_data.get_img_dataset import get_img_dataset
from get_data.get_pat_dataset import get_pat_dataset
from get_data.preprocess_data import preprocess_data




if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/GitHub_Test'
    data_dir = os.path.join(proj_dir, 'raw_image/HN')
    data_reg_dir = os.path.join(proj_dir, 'reg_image/HN')
    label_dir = os.path.join(proj_dir, 'label')
    out_dir = os.path.join(proj_dir, 'output')
    pro_data_dir = os.path.join(pro_data_dir, 'pro_data')

    if not os.path.exists(data_dir):
        os.path.mkdir(data_dir)
    if not os.path.exists(data_reg_dir):
        os.path.mkdir(data_reg_dir)
    if not os.path.exists(label_dir):
        os.path.mkdir(label_dir)
    if not os.path.exists(out_dir):
        os.path.mkdir(out_dir)
    if not os.path.exists(pro_data_dir):
        os.path.mkdir(pro_data_dir)

    print('\n--- STEP 1 - GET DATA ---\n')

    preprocess_data(
        data_dir=data_dir,
        data_reg_dir=data_reg_dir,
        new_spacing=(1, 1, 3),
        data_exclude=None,
        crop_shape=[192, 192, 100]
        )

    data_tot, label_tot, ID_tot = get_pat_dataset(
        data_dir=data_dir,
        data_reg_dir=data_reg_dir,
        label_dir=label_dir,
        pro_data_dir=pro_data_dir,
        )

    get_img_dataset(
        pro_data_dir=pro_data_dir,
        run_type=run_type,
        data_tot=data_tot,
        ID_tot=ID_tot,
        label_tot=label_tot,
        slice_range=range(17, 83),
        )



