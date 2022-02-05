
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
from go_model.finetune_model import finetune_model
from utils.write_txt import write_txt
from utils.get_stats_plots import get_stats_plots
from go_model.evaluate_model import evaluate_model
from get_data.exval_dataset import exval_pat_dataset
from get_data.exval_dataset import exval_img_dataset


  
if __name__ == '__main__':


    out_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection'
    proj_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project' 
    batch_size = 32
    lr = 1e-5
    activation = 'sigmoid'
    freeze_layer = None
    epoch = 1
    get_data = False
    fine_tune = False
    HN_modiel = 'EffNet_2021_08_21_22_41_34'
    saved_model = 'Tuned_EfficientNetB4_2021_08_27_20_26_55'
    run_model = 'EffNetB4'    
    
    print('\n--- STEP 4 - MODEL EX VAL ---\n')   
    
    # get chest CT data anddo  preprocessing
    if get_data == True:
        exval_pat_dataset(
            out_dir=out_dir,
            proj_dir=proj_dir, 
            crop_shape=[192, 192, 140],
            )

        exval_img_dataset(
            proj_dir=proj_dir,
            slice_range=range(50, 120),
            )

    ## fine tune models by freezing some layers
    if fine_tune == True:
        tuned_model, model_fn = finetune_model(
            out_dir=out_dir, 
            proj_dir=proj_dir, 
            HN_model=HN_model,
            batch_size=batch_size, 
            epoch=epoch, 
            freeze_layer=freeze_layer,           
            )
    
    ## evaluate finetuned model on chest CT data
    for run_type in ['exval1', 'exval2']:
        loss, acc = evaluate_model(
            run_type=run_type,
            out_dir=out_dir,
            proj_dir=proj_dir,
            saved_model=saved_model,
            )
        get_stats_plots(
            out_dir=out_dir,
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


    
