
import os
import numpy as np
import pandas as pd
import glob
import yaml
import argparse
from time import gmtime, strftime
from datetime import datetime
import timeit
from prediction.data_prepro import data_prepro
from prediction.model_pred import model_pred




if __name__ == '__main__':

    """
    Prediction on IV contrast for HeadNeck and Chest CT scan;
    
    Args:
        proj_dir {path} -- main folder to store models, data and outputs;
        body_part {str} -- choose either 'HeadNeck' or 'Chest' to run prediction;
    
    Return:
        Dataframe on contrast predicions on image and scan levels;
    
    """    
    

    body_part = 'Chest'
    
    proj_dir = '/path/to/my/folder'
    model_dir = os.path.join(proj_dir, 'models')
    data_dir = os.path.join(proj_dir, 'datasets')
    out_dir = os.path.join(proj_dir, 'outputs')
    
    os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
    os.mkdir(data_dir) if not os.path.isdir(data_dir) else None
    os.mkdir(out_dir) if not os.path.isdir(out_dir) else None
    
    # choose image for registration template 
    if body_part == 'HeadNeck':
        reg_temp_img = 'PMH049.nrrd'
    elif body_part == 'Chest':
        reg_temp_img = 'H570.nrrd'    

    print('\n--- STEP 5 - MODEL PREDICTION ---\n')
    
    # data preprocessing
    df_img, img_arr = data_prepro(
        body_part=body_part,
        data_dir=data_dir,
        reg_temp_img=reg_temp_img
        )

    # model prediction
    model_pred(
        body_part=body_part,
        model_dir=model_dir,
        out_dir=out_dir,
        df_img=df_img,
        img_arr=img_arr
        )
    
    print('model prediction done!')
