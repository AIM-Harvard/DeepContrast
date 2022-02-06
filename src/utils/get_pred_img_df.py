import pandas as pd
import numpy as np
import os
from utils.make_plots import make_plots

def get_pred_img_df(ahmed_data_dir, pro_data_dir, slice_range):
    
    df = pd.read_csv(os.path.join(ahmed_data_dir, 'rtog-0617_img_pred.csv'))
    df_label = pd.read_csv(os.path.join(pro_data_dir, 'label_RTOG0617.csv'))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    pat_label = df_label['Contrast'].to_list()
    img_label = []
    for label in pat_label:
        list1 = [label] * slice_number
        img_label.extend(list1)
    df['label'] = img_label
    df.to_csv(os.path.join(pro_data_dir, 'exval2_img_pred.csv'))

if __name__ == '__main__':
    
    ahmed_data_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro'
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    exval2_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/d_pro'
    slice_range = range(50, 70)
    saved_model = 'FineTuned_model_2021_07_27_16_44_40'

    get_pred_img_df(
        ahmed_data_dir=ahmed_data_dir,
        pro_data_dir=pro_data_dir,
        slice_range=slice_range
        )

    make_plots(
        run_type='exval2',
        thr_img=0.321,
        thr_prob=0.379,
        thr_pos=0.575,
        bootstrap=1000,
        pro_data_dir=pro_data_dir,
        save_dir=exval2_dir,
        loss=None,
        acc=None,
        run_model='ResNet',
        saved_model=saved_model,
        epoch=100,
        batch_size=32,
        lr=1e-5
        )
