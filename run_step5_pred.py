
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
    
    body_part = 'head_and_neck'
    out_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection'
    reg_temp_img = 'PMH049.nrrd'

    print('\n--- STEP 5 - MODEL PREDICTION ---\n')
    
    # data preprocessing
    df_img, img_arr = data_prepro(
        body_part=body_part,
        out_dir=out_dir,
        reg_temp_img=reg_temp_img
        )

    # model prediction
    model_pred(
        body_part=body_part,
        df_img=df_img,
        img_arr=img_arr,
        out_dir=out_dir,
        )

