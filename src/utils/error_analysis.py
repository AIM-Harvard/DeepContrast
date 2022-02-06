
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
from collections import Counter
from datetime import datetime
from time import localtime, strftime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from crop_image import crop_image
from resize_3d import resize_3d
import SimpleITK as sitk

#----------------------------------------------------------------
# error analysis on image level
#---------------------------------------------------------------
def error_img(run_type, val_save_dir, test_save_dir, input_channel, crop, pro_data_dir):
   
    ### load train data based on input channels
    if run_type == 'val': 
        file_dir = val_save_dir
        input_fn = 'df_val_pred.csv'
        output_fn = 'val_error_img.csv'
        col = 'y_val'
    elif run_type == 'test':
        file_dir = test_save_dir
        input_fn = 'df_test_pred.csv'
        output_fn = 'test_error_img.csv'
        col = 'y_test'
    elif run_type == 'exval2':
        file_dir = pro_data_dir
        input_fn = 'exval2_img_pred.csv'
        output_fn = 'exval2_error_img.csv'
        col = 'y_exval2'

    ### dataframe with label and predictions
    df = pd.read_csv(os.path.join(file_dir, input_fn))
    print(df[0:10])
    #df.drop([col, 'ID'], axis=1, inplace=True)
    df = df.drop(df[df['y_pred_class'] == df['label']].index)
    df[['y_pred']] = df[['y_pred']].round(3)
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    print(df[0:200]) 
    df.to_csv(os.path.join(file_dir, output_fn))
    df_error_img = df

    return df_error_img
#----------------------------------------------------
# generate images for error check
#---------------------------------------------------- 
def save_error_img(df_error_img, run_type, n_img, val_img_dir, test_img_dir, 
                   val_error_dir, test_error_dir, pro_data_dir):

    ### store all indices that has wrong predictions
    indices = []
    df = df_error_img
    for i in range(df.shape[0]):
        indices.append(i)
    print(x_val.shape)
    arr = x_val.take(indices=indices, axis=0)
    print(arr.shape)
    arr = arr[:, :, :, 0]
    print(arr.shape)
    arr = arr.reshape((arr.shape[0], 192, 192))
    print(arr.shape)
    
    ## load img data
    if run_type == 'val':
        if input_channel == 1:
            fn = 'val_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'val_arr_3ch.npy'
        x_val = np.load(os.path.join(data_pro_dir, fn))
        file_dir = val_error_dir
    elif run_type == 'test':
        if input_channel == 1:
            fn = 'test_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'test_arr_3ch.npy'
        x_val = np.load(os.path.join(data_pro_dir, fn))
        file_dir = test_error_dir
    elif run_type == 'exval2':
        if input_channel == 1:
            fn = 'exval2_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'exval2_arr.npy'
        x_val = np.load(os.path.join(pro_data_dir, fn))
        file_dir = pro_data_dir

    ### display images for error checks
    count = 0
    for i in range(n_img):
    # for i in range(arr.shape[0]):
        #count += 1
        #print(count)
        img = arr[i, :, :]
        fn = str(i) + '.jpeg'
        img_fn = os.path.join(file_dir, fn)
        mpl.image.imsave(img_fn, img, cmap='gray')
     
    print('save images complete!!!')

#----------------------------------------------------------------------
# error analysis on patient level
#----------------------------------------------------------------------
def error_pat(run_type, val_dir, test_dir, exval2_dir, threshold, pro_data_dir):

    if run_type == 'val':
        input_fn = 'val_img_pred.csv'
        output_fn = 'val_pat_error.csv'
        drop_col = 'y_val'
        save_dir = val_dir
    elif run_type == 'test':
        input_fn = 'test_img_pred.csv'
        output_fn = 'test_pat_error.csv'
        drop_col = 'y_test'
        save_dir = test_dir
    elif run_type == 'exval2':
        input_fn = 'rtog_img_pred.csv'
        output_fn = 'rtog_pat_error.csv'
        drop_col = 'y_exval2'
        save_dir = exval2_dir

    df_sum = pd.read_csv(os.path.join(pro_data_dir, input_fn))
    df_mean = df_sum.groupby(['ID']).mean()
    y_true = df_mean['label']
    y_pred = df_mean['y_pred']
    y_pred_classes = []
    for pred in y_pred:
        if pred < threshold:
            y_pred_class = 0
        elif pred >= threshold:
            y_pred_class = 1
        y_pred_classes.append(y_pred_class)
    df_mean['y_pred_class_thr'] = y_pred_classes
    df = df_mean
    df[['y_pred', 'y_pred_class']] = df[['y_pred', 'y_pred_class']].round(3)
    df['label'] = df['label'].astype(int)
    df = df.drop(df[df['y_pred_class_thr'] == df['label']].index)
    #df.drop(['y_exval'], inplace=True, axis=1)
    df.drop([drop_col], inplace=True, axis=1)
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    print(df)
    #print("miss predicted scan:", df.shape[0])
    df_error_patient = df
    df.to_csv(os.path.join(save_dir, output_fn))

#----------------------------------------------------------------------
# save error scan
#----------------------------------------------------------------------    
def save_error_pat(run_type, val_save_dir, test_save_dir, val_error_dir, test_error_dir, norm_type,
                   interp_type, output_size, PMH_reg_dir, CHUM_reg_dir, CHUS_reg_dir):
    count = 0
    dirs = []
    
    if run_type == 'val':
        save_dir = val_error_dir
        df = pd.read_csv(os.path.join(val_save_dir, 'val_error_patient_new.csv'))
        IDs = df['ID'].to_list()
        for ID in IDs:
            print('error scan:', ID)
            fn = str(ID) + '.nrrd'
            if ID[:-3] == 'PMH':
                dir = os.path.join(PMH_reg_dir, fn)
            elif ID[:-3] == 'CHUS':
                dir = os.path.join(CHUS_reg_dir, fn)
            elif ID[:-3] == 'CHUM':
                dir = os.path.join(CHUM_reg_dir, fn)
            dirs.append(dir)
   
    elif run_type == 'test':
        save_dir = test_error_dir
        df = pd.read_csv(os.path.join(test_save_dir, 'test_error_patient_new.csv')) 
        IDs = df['ID'].to_list()
        for ID in IDs:
            print('error scan:', ID)
            fn = str(ID) + '.nrrd'
            dir = os.path.join(MDACC_reg_dir, fn)
            dirs.append(dir)

    for file_dir, patient_id in zip(dirs, IDs):
        count += 1
        print(count)
        nrrd = sitk.ReadImage(file_dir, sitk.sitkFloat32)
        img_arr = sitk.GetArrayFromImage(nrrd)
        data = img_arr[30:78, :, :]
        data[data <= -1024] = -1024
        data[data > 700] = 0
        if norm_type == 'np_interp':
            arr_img = np.interp(data, [-200, 200], [0, 1])
        elif norm_type == 'np_clip':
            arr_img = np.clip(data, a_min=-200, a_max=200)
            MAX, MIN = arr_img.max(), arr_img.min()
            arr_img = (arr_img - MIN) / (MAX - MIN)
        ## save npy array to image 
        img = sitk.GetImageFromArray(arr_img)
        fn = str(patient_id) + '.nrrd'
        sitk.WriteImage(img, os.path.join(save_dir, fn))
    print('save error scans!')

    ## save error scan as nrrd file
        
#----------------------------------------------------------------------------
# main funtion
#---------------------------------------------------------------------------
if __name__ == '__main__':

    val_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/val'
    test_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/test'
    exval2_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval2'
    val_error_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/val/error'
    test_error_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/test/error'
    mdacc_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_mdacc'
    CHUM_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/CHUM_data_reg'
    CHUS_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/CHUS_data_reg'
    PMH_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/PMH_data_reg'
    MDACC_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/MDACC_data_reg'
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    
    input_channel = 3
    crop = True
    n_img = 20
    save_img = False
    save_pat = False
    error_image = False
    thr_prob = 0.5
    run_type = 'val'
    norm_type = 'np_clip'
    crop_shape = [192, 192, 110]
    return_type1 = 'nrrd'
    return_type2 = 'npy'
    interp_type = 'linear'
    output_size = (96, 96, 36)

    error_pat(
        run_type=run_type,
        val_dir=val_dir,
        test_dir=test_dir,
        exval2_dir=exval2_dir,
        threshold=thr_prob,
        pro_data_dir=pro_data_dir
        )

    if error_image == True:
        df = error_img(
            run_type=run_type,
            val_save_dir=val_save_dir,
            test_save_dir=test_save_dir,
            input_channel=input_channel,
            crop=crop,
            pro_data_dir=pro_data_dir
            )
 
    if save_pat == True:
        save_error_pat(
            run_type=run_type, 
            val_save_dir=val_save_dir,
            test_save_dir=test_save_dir,
            val_error_dir=val_error_dir, 
            test_error_dir=test_error_dir, 
            norm_type=norm_type,
            interp_type=interp_type, 
            output_size=output_size,
            PMH_reg_dir=PMH_reg_dir,
            CHUM_reg_dir=CHUM_reg_dir,
            CHUS_reg_dir=CHUS_reg_dir
            )
    
    if save_img == True:
        save_error_img(
            df=df,
            n_img=n_img,
            run_type=run_type,
            val_img_dir=val_img_dir,
            test_img_dir=test_img_dir,
            val_error_dir=vel_error_dir,
            test_error_dir=test_error_dir,
            pro_data_dir=pro_data_dir
            )
