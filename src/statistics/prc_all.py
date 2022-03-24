import os
import numpy as np
import pandas as pd
import pickle
from statistics.plot_prc import plot_prc



def prc_all(run_type, level, thr_prob, thr_pos, color, pro_data_dir, save_dir, fn_df_pred):

    df_sum = pd.read_csv(os.path.join(pro_data_dir, fn_df_pred))

    if level == 'img':
        y_true = df_sum['label'].to_numpy()
        y_pred = df_sum['y_pred'].to_numpy()
        print_info = 'prc image:'
    elif level == 'patient_mean_prob':
        df_mean = df_sum.groupby(['ID']).mean()
        y_true = df_mean['label'].to_numpy()
        y_pred = df_mean['y_pred'].to_numpy()
        print_info = 'prc patient prob:'
    elif level == 'patient_mean_pos':
        df_mean = df_sum.groupby(['ID']).mean()
        y_true = df_mean['label'].to_numpy()
        y_pred = df_mean['y_pred_class'].to_numpy()
        print_info = 'prc patient pos:'
   
    prc_auc = plot_prc(
        save_dir=save_dir,
        y_true=y_true,
        y_pred=y_pred,
        level=level,
        color=color,
        )

    print(print_info)
    print(prc_auc)

    return prc_auc

