
"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step5
  ----------------------------------------------
  ----------------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.8.5
  ----------------------------------------------
  
"""

import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import nrrd
import scipy.stats as ss
import SimpleITK as stik
import glob
from PIL import Image
from collections import Counter
import skimage.transform as st
from datetime import datetime
from time import gmtime, strftime
import pickle
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




def pred_model(model_dir, pred_data_dir, saved_model, thr_img, thr_prob, fns_pat_pred,
               fns_img_pred, fns_arr, fns_img_df):    

    """
    model prediction

    @params:
      model_dir     - required : path to load CNN model
      pro_data_dir  - required : path to folder that saves all processed data
      saved_model   - required : CNN model name from training step
      input_channel - required : 1 or 3, usually 3
      threshold     - required : threshold to decide predicted label

    """

    for fn_arr, fn_img_df, fn_img_pred in zip(fns_arr, fns_img_df, fns_img_pred):

        ### load numpy array
        x_exval = np.load(os.path.join(pred_data_dir, fn_arr))
        ### load ID
        df = pd.read_csv(os.path.join(pred_data_dir, fn_img_df))
        ### load saved model
        model = load_model(os.path.join(model_dir, saved_model))
        ## prediction
        y_pred = model.predict(x_exval)
        y_pred_class = [1 * (x[0] >= thr_img) for x in y_pred]

        ### save a dataframe for test and prediction
        ID = []
        for file in df['fn']:
            id = file.split('\\')[-1].split('_s')[0].strip()
            ID.append(id)
        df['ID'] = ID
        df['y_pred'] = y_pred
        df['y_pred_class'] = y_pred_class
        df_img_pred = df[['ID', 'fn', 'y_pred', 'y_pred_class']] 
        df_img_pred.to_csv(os.path.join(pred_data_dir, fn_img_pred), index=False) 
    
    ## calcualte patient level prediction
    for fn_img_pred, fn_pat_pred in zip(fns_img_pred, fns_pat_pred):
        df = pd.read_csv(os.path.join(pred_data_dir, fn_img_pred))
        df.drop(['fn'], axis=1, inplace=True)
        df_mean = df.groupby(['ID']).mean()
        preds = df_mean['y_pred']
        y_pred = []
        for pred in preds:
            if pred > thr_prob:
                pred = 1
            else:
                pred = 0
            y_pred.append(pred)
        df_mean['predictions'] = y_pred
        df_mean.drop(['y_pred', 'y_pred_class'], axis=1, inplace=True)
        df_mean.to_csv(os.path.join(pred_data_dir, fn_pat_pred))
        print(str(fn_pat_pred.split('_')[0]) + str(' cohort'))
        print('Total scan:', df_mean.shape[0])
        print(df_mean['predictions'].value_counts())



    

    
