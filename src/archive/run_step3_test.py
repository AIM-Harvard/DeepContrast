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
from tensorflow.keras.optimizers import Adam
from go_model.evaluate_model import evaluate_model
from utils.write_txt import write_txt
from utils.get_stats_plots import get_stats_plots




if __name__ == '__main__':

    
    proj_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project'
    out_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection'
    saved_model = 'EffNet_2021_08_21_22_41_34'
    # 'ResNet_2021_07_18_06_28_40', 'cnn_2021_07_19_21_56_34', 'inception_2021_08_21_15_16_12' 
    batch_size = 32
    epoch = 500
    lr = 1e-5
    run_model = 'EffNet'
    
    model_dir = os.path.join(out_dir, 'model')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(pro_data_dir):
        os.mkdir(pro_data_dir)
    
    print('\n--- STEP 3 - MODEL EVALUATION ---\n')   
    
    for run_type in ['val', 'test']:
        # evalute model
        loss, acc = evaluate_model(
            run_type=run_type,
            model_dir=model_dir,
            pro_data_dir=pro_data_dir,
            saved_model=saved_model,
            threshold=0.5,
            activation='sigmoid'
            )
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
            bootstrap=1000,
            )
