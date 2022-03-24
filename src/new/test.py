
import os
import timeit
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




def evaluate_model(run_type, model_dir, pro_data_dir, saved_model, 
                   threshold=0.5, activation='sigmoid'):    
    
    """
    Evaluate model for validation/test/external validation data;

    Args:
        out_dir {path} -- path to main output folder;
        proj_dir {path} -- path to main project folder;
        saved_model {str} -- saved model name;
        tuned_model {Keras model} -- finetuned model for chest CT;
    
    Keyword args:
        threshold {float} -- threshold to determine postive class;
        activation {str or function} -- activation function, default: 'sigmoid';
    
    Returns:
        training accuracy, loss, model
    
    """


    # load data and label based on run type
    #--------------------------------------- 
    if run_type == 'val':
        fn_data = 'val_arr_3ch.npy'
        fn_label = 'val_img_df.csv'
        fn_pred = 'val_img_pred.csv'
    elif run_type == 'test':
        fn_data = 'test_arr_3ch.npy'
        fn_label = 'test_img_df.csv'
        fn_pred = 'test_img_pred.csv'
    elif run_type == 'exval1':
        fn_data = 'exval1_arr2.npy'
        fn_label = 'exval1_img_df2.csv'
        fn_pred = 'exval1_img_pred.csv'
    elif run_type == 'exval2':
        fn_data = 'rtog_0617_arr.npy'
        fn_label = 'rtog_img_df.csv'
        fn_pred = 'exval2_img_pred.csv'
    
    x_data = np.load(os.path.join(pro_data_dir, fn_data))
    df = pd.read_csv(os.path.join(pro_data_dir, fn_label))
    y_label = np.asarray(df['label']).astype('int').reshape((-1, 1))

    ## load saved model and evaluate
    #-------------------------------
    model = load_model(os.path.join(model_dir, saved_model))
    y_pred = model.predict(x_data)
    score = model.evaluate(x_data, y_label)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('loss:', loss)
    print('acc:', acc)
    
    if activation == 'sigmoid':
        y_pred = model.predict(x_data)
        y_pred_class = [1 * (x[0] >= threshold) for x in y_pred]
    elif activation == 'softmax':
        y_pred_prob = model.predict(x_data)
        y_pred = y_pred_prob[:, 1]
        y_pred_class = np.argmax(y_pred_prob, axis=1)

    # save a dataframe for prediction
    #----------------------------------
    ID = []
    for file in df['fn']:
        if run_type in ['val', 'test', 'exval1']:
            id = file.split('\\')[-1].split('_')[0].strip()
        elif run_type == 'exval2':
            id = file.split('\\')[-1].split('_s')[0].strip()
        ID.append(id)
    df['ID'] = ID
    df['y_pred'] = y_pred
    df['y_pred_class'] = y_pred_class
    df_test_pred = df[['ID', 'fn', 'label', 'y_pred', 'y_pred_class']]
    df_test_pred.to_csv(os.path.join(pro_data_dir, fn_pred)) 
    
    return loss, acc

        



    

    
