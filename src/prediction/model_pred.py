
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



def model_pred(body_part, save_csv, model_dir, out_dir, df_img, img_arr, thr_img=0.5, thr_pat=0.5):    

    """
    model prediction for IV contrast

    Arguments:
        body_part {str} -- 'HeadNeck' or 'Chest'
        df_img {pd.df} -- dataframe with scan and axial slice ID.
        img_arr {np.array} -- numpy array stacked with axial image slices.
        model_dir {str} -- directory for saved model.
        out_dir {str} -- directory for results output.

    Keyword arguments:
        thr_img {float} -- threshold to determine prediction class on image level.
        thr_pat {float} -- threshold to determine prediction class on patient level. 

    return:
        dataframes of model predictions on image level and patient level
    """
    

    if body_part == 'HeadNeck':
        saved_model = 'EffNet_HeadNeck.h5'
    elif body_part == 'Chest':
        saved_model = 'EffNet_Chest.h5'

    ## load saved model
    #print(str(saved_model))
    model = load_model(os.path.join(model_dir, saved_model))
    ## prediction
    y_pred = model.predict(img_arr, batch_size=32)
    y_pred_class = [1 * (x[0] >= thr_img) for x in y_pred]

    ## save a dataframe for prediction on image level
    df_img['y_pred'] = np.around(y_pred, 3)
    df_img['y_pred_class'] = y_pred_class
    df_img_pred = df_img[['pat_id', 'img_id', 'y_pred', 'y_pred_class']] 
    if save_csv:
        fn = 'image_prediction' + '.csv'
        df_img_pred.to_csv(os.path.join(out_dir, fn), index=False) 
        print('Saved image prediction!')

    ## calcualte patient level prediction
    df_img_pred.drop(['img_id'], axis=1, inplace=True)
    df_mean = df_img_pred.groupby(['pat_id']).mean().reset_index()
    preds = df_mean['y_pred']
    y_pred = []
    for pred in preds:
        if pred > thr_pat:
            pred = 1
        else:
            pred = 0
        y_pred.append(pred)
    df_mean['predictions'] = y_pred
    df_mean.drop(['y_pred', 'y_pred_class'], axis=1, inplace=True)
    df_pat_pred = df_mean 
    print('patient level pred:\n', df_pat_pred)
    if save_csv:
        fn = 'patient_prediction' + '.csv'
        df_pat_pred.to_csv(os.path.join(out_dir, fn))
        print('Saved patient prediction!')


    

    
