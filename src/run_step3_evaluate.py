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
from statistics.write_txt import write_txt
from statistics.get_stats_plots import get_stats_plots




if __name__ == '__main__':

    
    proj_dir = 'root path'
    out_dir = os.path.join(proj_dir, 'HeadNeck/output')
    pro_data_dir = os.path.join(proj_dir, 'HeadNeck/pro_data')
    model_dir = os.path.join(proj_dir, 'HeadNeck/model')
    batch_size = 32
    epoch = 500
    lr = 1e-5
    run_model = 'EffNetB4'
    saved_model = 'EffNetB4'
     
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
            pro_data_dir=pro_data_dir,
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
