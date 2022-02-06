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
from get_data.get_img_dataset import get_img_dataset
from get_data.get_pat_dataset import get_pat_dataset
from get_data.preprocess_data import preprocess_data




if __name__ == '__main__':

    proj_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project'    
    data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST'
    out_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection'
    data_exclude = None

    print('\n--- STEP 1 - GET DATA ---\n')

    preprocess_data(
        data_dir=data_dir,
        out_dir=out_dir,
        new_spacing=(1, 1, 3),
        data_exclude=None,
        crop_shape=[192, 192, 100]
        )

    data_tot, label_tot, ID_tot = get_pat_dataset(
        data_dir=data_dir,
        out_dir=out_dir,
        proj_dir=proj_dir,
        )

    get_img_dataset(
        proj_dir=proj_dir,
        run_type=run_type,
        data_tot=data_tot,
        ID_tot=ID_tot,
        label_tot=label_tot,
        slice_range=range(17, 83),
        )
