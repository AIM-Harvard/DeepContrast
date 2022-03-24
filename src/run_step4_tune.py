
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
from go_model.tune_model import tune_model
from statistics.write_txt import write_txt
from statistics.get_stats_plots import get_stats_plots
from go_model.evaluate_model import evaluate_model
from train_data.tune_dataset import tune_pat_dataset
from train_data.tune_dataset import tune_img_dataset


  
if __name__ == '__main__':

    proj_dir = 'root path'  
    data_dir = os.path.join(proj_dir, 'Chest/raw_image')
    pre_data_dir = os.path.join(proj_dir, 'Chest/pre_image')
    label_dir = os.path.join(proj_dir, 'Chest/label')
    out_dir = os.path.join(proj_dir, 'Chest/output')
    pro_data_dir = os.path.join(proj_dir, 'Chest/pro_data')
    model_dir = os.path.join(proj_dir, 'Chest/model')

    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    if not os.path.exists(pre_data_dir):                
        os.makedirs(pre_data_dir)
    if not os.path.exists(pro_data_dir):                
        os.makedirs(pro_data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    label_file = 'CH_label.csv'
    batch_size = 32
    lr = 1e-5
    activation = 'sigmoid'
    freeze_layer = None
    epoch = 1
    get_data = False
    fine_tune = False
    HN_model = 'EffNetB4'
    saved_model = 'Tuned_EffNetB4'
    run_model = 'EffNetB4'    
    
    print('\n--- STEP 4 - MODEL FINE TUNE ---\n')   
    
    # get chest CT data anddo  preprocessing
    if get_data == True:
        tune_pat_dataset(
            data_dir=data_dir,
            pre_data_dir=pre_data_dir,
            pro_data_dir=pro_data_dir,
            label_dir=label_dir,
            label_file=label_file,
            crop_shape=[192, 192, 140],
            )
        tune_img_dataset(
            pro_data_dir=pro_data_dir,
            pre_data_dir=pre_data_dir,
            slice_range=range(50, 120),
            )

    ## fine tune models by freezing some layers
    if fine_tune == True:
        tuned_model, model_fn = tune_model(
            model_dir=model_dir, 
            pro_data_dir=pro_data_dir, 
            HN_model=HN_model,
            batch_size=batch_size, 
            epoch=epoch, 
            freeze_layer=freeze_layer,           
            )
    
    ## evaluate finetuned model on chest CT data
    for run_type in ['tune']:
        loss, acc = evaluate_model(
            run_type=run_type,
            model_dir=model_dir,
            pro_data_dir=pro_data_dir,
            saved_model=saved_model,
            )
        get_stats_plots(
            pro_data_dir=pro_data_dir,
            proj_dir=proj_dir,
            run_type=run_type,
            run_model=run_model,
            loss=loss,
            acc=acc,
            saved_model=saved_model,
            epoch=epoch,
            batch_size=batch_size,
            lr=lr
            )


    
