import os
import numpy as np
import pandas as pd
import pickle
from time import gmtime, strftime
from datetime import datetime
import timeit
from statistics.cm_all import cm_all
from statistics.roc_all import roc_all
from statistics.prc_all import prc_all
#from utils.acc_loss import acc_loss
from statistics.write_txt import write_txt



def get_stats_plots(pro_data_dir, root_dir, run_type, run_model, loss, acc, 
                    saved_model, epoch, batch_size, lr, thr_img=0.5, 
                    thr_prob=0.5, thr_pos=0.5, bootstrap=1000):

    """
    generate model val/test statistics and plot curves;

    Args:
        loss {float} -- validation loss;
        acc {float} -- validation accuracy;
        run_model {str} -- cnn model name;
        batch_size {int} -- batch size for data loading;
        epoch {int} -- training epoch;
        out_dir {path} -- path for output files;
        opt {str or function} -- optimized function: 'adam';
        lr {float} -- learning rate;
    
    Keyword args:
        bootstrap {int} -- number of bootstrap to calculate 95% CI for AUC;
        thr_img {float} -- threshold to determine positive class on image level;
        thr_prob {float} -- threshold to determine positive class on patient 
                            level (mean prob score);
        thr_pos {float} -- threshold to determine positive class on patient 
                           level (positive class percentage);
    Returns:
       Model prediction statistics and plots: ROC, PRC, confusion matrix, etc.
    
    """
    
    train_dir = os.path.join(root_dir, 'HeadNeck/output/train')
    val_dir = os.path.join(root_dir, 'HeadNeck/output/val')
    test_dir = os.path.join(root_dir, 'HeadNeck/output/test')
    tune_dir = os.path.join(root_dir, 'Chest/output/tune')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(tune_dir):
        os.makedirs(tune_dir)

    ### determine if this is train or test
    if run_type == 'val':
        fn_df_pred = 'val_img_pred.csv'
        save_dir = val_dir
    elif run_type == 'test':
        fn_df_pred = 'test_img_pred.csv'
        save_dir = test_dir
    elif run_type == 'tune':
        fn_df_pred = 'test_img_pred.csv'
        save_dir = tune_dir

    cms = []
    cm_norms = []
    reports = []
    roc_stats = []
    prc_aucs = []
    levels = ['img', 'patient_mean_prob', 'patient_mean_pos']

    for level in levels:
        
        ## confusion matrix
        cm, cm_norm, report = cm_all(
            run_type=run_type,
            level=level,
            thr_img=thr_img,
            thr_prob=thr_prob,
            thr_pos=thr_pos,
            pro_data_dir=pro_data_dir,
            save_dir=save_dir,
            fn_df_pred=fn_df_pred
            )
        cms.append(cm)
        cm_norms.append(cm_norm)
        reports.append(report)

        ## ROC curves
        roc_stat = roc_all(
            run_type=run_type,
            level=level,
            thr_prob=thr_prob,
            thr_pos=thr_pos,
            bootstrap=bootstrap,
            color='blue',
            pro_data_dir=pro_data_dir,
            save_dir=save_dir,
            fn_df_pred=fn_df_pred
            )
        roc_stats.append(roc_stat)

        ## PRC curves
        prc_auc = prc_all(
            run_type=run_type,
            level=level,
            thr_prob=thr_prob,
            thr_pos=thr_pos,
            color='red',
            pro_data_dir=pro_data_dir,
            save_dir=save_dir,
            fn_df_pred=fn_df_pred
            )
        prc_aucs.append(prc_auc)

    ### save validation results to txt
    write_txt(
        run_type=run_type,
        root_dir=root_dir,
        loss=loss,
        acc=acc,
        cms=cms,
        cm_norms=cm_norms,
        reports=reports,
        prc_aucs=prc_aucs,
        roc_stats=roc_stats,
        run_model=run_model,
        saved_model=saved_model,
        epoch=epoch,
        batch_size=batch_size,
        lr=lr
        )

    print('saved model as:', saved_model)

