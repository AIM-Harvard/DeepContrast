
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
        body_part {str} -- choose either 'HeadNeck' or 'Chest' to run prediction;
    
    Return:
        Dataframe on contrast predicions on image and scan levels;
    
    """    
    

    body_part = 'HeadNeck'
    
    proj_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    model_dir = os.path.join(proj_dir, 'models')
    data_dir = os.path.join(proj_dir, 'datasets') 

    print('\n--- STEP 5 - MODEL PREDICTION ---\n')
    
    # data preprocessing
    df_img, img_arr = data_prepro(
        body_part=body_part,
        data_dir=data_dir,
        )

    # model prediction
    model_pred(
        body_part=body_part,
        model_dir=model_dir,
        out_dir=proj_dir,
        df_img=df_img,
        img_arr=img_arr
        )
    
    print('model prediction done!')
