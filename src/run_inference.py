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


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--body_part',
        type = str,
        choices = ['HeadNeck', 'Chest'],
        help = 'Specify the body part ("HeadNeck" or "Chest") to run inference.'
        )
    parser.add_argument(
        '-v',
        '--save_csv',
        required = False,
        help = 'Specify True or False to save csv for contrast prediction.',
        default = 'False'
        )
    args = parser.parse_args()
    
    proj_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    model_dir = os.path.join(proj_dir, 'models')
    data_dir = os.path.join(proj_dir, 'datasets') 

    print('\n--- MODEL INFERENCE ---\n')
    
    # data preprocessing
    df_img, img_arr = data_prepro(
        body_part=args.body_part,
        data_dir=data_dir,
        )

    # model prediction
    model_pred(
        body_part=args.body_part,
        save_csv=args.save_csv,
        model_dir=model_dir,
        out_dir=proj_dir,
        df_img=df_img,
        img_arr=img_arr
        )
    
    print('Model prediction done!')
