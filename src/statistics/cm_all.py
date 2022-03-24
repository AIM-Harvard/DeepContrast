import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from statistics.plot_cm import plot_cm



def cm_all(run_type, level, thr_img, thr_prob, thr_pos, pro_data_dir, save_dir, fn_df_pred):
    
    df_sum = pd.read_csv(os.path.join(pro_data_dir, fn_df_pred))

    if level == 'img':
        y_true = df_sum['label'].to_numpy()
        preds = df_sum['y_pred'].to_numpy()
        y_pred = []
        for pred in preds:
            if pred > thr_img:
                pred = 1
            else:
                pred = 0
            y_pred.append(pred)
        y_pred = np.asarray(y_pred)
        print_info = 'cm image:'

    elif level == 'patient_mean_prob': 
        df_mean = df_sum.groupby(['ID']).mean()
        y_true = df_mean['label'].to_numpy()
        preds = df_mean['y_pred'].to_numpy()
        y_pred = []
        for pred in preds:
            if pred > thr_prob:
                pred = 1
            else:
                pred = 0
            y_pred.append(pred)
        y_pred = np.asarray(y_pred)
        print_info = 'cm patient prob:'

    elif level == 'patient_mean_pos':
        df_mean = df_sum.groupby(['ID']).mean()
        y_true = df_mean['label'].to_numpy()
        pos_rates = df_mean['y_pred_class'].to_list()
        y_pred = []
        for pos_rate in pos_rates:
            if pos_rate > thr_pos:
                pred = 1
            else:
                pred = 0
            y_pred.append(pred)
        y_pred = np.asarray(y_pred)
        print_info = 'cm patient pos:'

    ### using confusion matrix to calculate AUC
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)
    
    ## classification report
    report = classification_report(y_true, y_pred, digits=3)

    # statistics
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    tn = cm[0][0]
    acc = (tp + tn)/(tp + fp + fn + tn)
    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)
    tpr = np.around(tpr, 3)
    tnr = np.around(tnr, 3)
    auc5 = (tpr + tnr)/2
    auc5 = np.around(auc5, 3)
    
    print(print_info)
    print(cm)
    print(cm_norm)
    print(report)
    
    ## plot cm
    for cm0, cm_type in zip([cm, cm_norm], ['raw', 'norm']):
        plot_cm(
            cm0=cm0,
            cm_type=cm_type,
            level=level,
            save_dir=save_dir
            )

    return cm, cm_norm, report





    

   
