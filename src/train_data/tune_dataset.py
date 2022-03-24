import glob
import shutil
import os
import pandas as pd
import numpy as np
import nrrd
import re
import matplotlib
import matplotlib.pyplot as plt
import pickle
from time import gmtime, strftime
from datetime import datetime
import timeit
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image
from utils.respacing import respacing
from utils.nrrd_reg import nrrd_reg_rigid_ref
from train_data.get_img_dataset import img_dataset



def tune_pat_dataset(data_dir, pre_data_dir, pro_data_dir, label_dir, label_file, 
                     crop_shape=[192, 192, 140], interp_type='linear', input_channel=3,  
                     norm_type='np_clip', data_exclude=None, new_spacing=[1, 1, 3]):
    
    """
    Preprocess data (respacing, registration, cropping) for chest CT dataset;

    Arguments:
        proj_dir {path} -- path to main project folder;
        out_dir {path} -- path to result outputs;
    
    Keyword arguments:
        new_spacing {tuple} -- respacing size, defaul [1, 1, 3];
        return_type {str} -- image data format after preprocessing, default: 'nrrd';
        data_exclude {str} -- exclude patient data due to data issue, default: None;
        crop_shape {np.array} -- numpy array size afer cropping;
        interp_type {str} -- interpolation type for respacing, default: 'linear';

    Return:
        save nrrd image data;
    """


    reg_temp_img = os.path.join(data_dir, 'CH001.nrrd')
    
    df_label = pd.read_csv(os.path.join(label_dir, label_file))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    labels = df_label['Contrast'].to_list()
    ## create df for dir, ID and labels on patient level
    fns = []
    IDs = []
    fns = [fn for fn in sorted(glob.glob(data_dir + '/*nrrd'))]
    for fn in sorted(glob.glob(data_dir + '/*nrrd')):
        ID = fn.split('/')[-1].split('.')[0].strip()
        IDs.append(ID)
    print('ID:', len(IDs))
    print('file:', len(fns))
    print('label:', len(labels))
    print('contrast scan in ex val:', labels.count(1))
    print('non-contrast scan in ex val:', labels.count(0))
    df = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    df.to_csv(os.path.join(pro_data_dir, 'tune_pat_df.csv'), index=False)
    print('total scan:', df.shape[0])
    
    ## delete excluded scans and repeated scans
    if data_exclude != None:
        df_exclude = df[df['ID'].isin(data_exclude)]
        print('exclude scans:', df_exclude)
        df.drop(df[df['ID'].isin(test_exclude)].index, inplace=True)
        print('total test scans:', df.shape[0])
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    #print(df[0:50])
   
    ### registration, respacing, cropping 
    for fn, ID in zip(df['file'], df['ID']):
        print(ID)
        ## respacing      
        img_nrrd = respacing(
            nrrd_dir=fn,
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type='nrrd',
            save_dir=None
            )
        ## registration
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_nrrd,
            fixed_img_dir=reg_temp_img,
            patient_id=ID,
            save_dir=None
            )
        ## crop image from (500, 500, 116) to (180, 180, 60)
        img_crop = crop_image(
            nrrd_file=img_reg,
            patient_id=ID,
            crop_shape=crop_shape,
            return_type='nrrd',
            save_dir=pre_data_dir
            )


def tune_img_dataset(pro_data_dir, pre_data_dir, slice_range=range(50, 120), input_channel=3, 
                     norm_type='np_clip', split=True, fn_arr_1ch=None):

    """
    get stacked image slices from scan level CT and corresponding labels and IDs;

    Args:
        run_type {str} -- train, val, test, external val, pred;
        pro_data_dir {path} -- path to processed data;
        nrrds {list} --  list of paths for CT scan files in nrrd format;
        IDs {list} -- list of patient ID;
        labels {list} -- list of patient labels;
        slice_range {np.array} -- image slice range in z direction for cropping;
        run_type {str} -- train, val, test, or external val;
        pro_data_dir {path} -- path to processed data;
        fn_arr_1ch {str} -- filename for 1 d numpy array for stacked image slices;
        fn_arr_3ch {str} -- filename for 3 d numpy array for stacked image slices;      
        fn_df {str} -- filename for dataframe contains image path, image labels and image ID;
    
    Keyword args:
        input_channel {str} -- image channel, default: 3;
        norm_type {str} -- image normalization type: 'np_clip' or 'np_linear';

    Returns:
        img_df {pd.df} -- dataframe contains preprocessed image paths, label, ID (image level);

    """

    df = pd.read_csv(os.path.join(pro_data_dir, 'tune_pat_df.csv'))
    fns = [fn for fn in sorted(glob.glob(pre_data_dir + '/*nrrd'))]
    labels = df['label']
    IDs = df['ID'] 
    ## split dataset for fine-tuning model and test model
    if split == True:
        data_train, data_test, label_train, label_test, ID_train, ID_test = train_test_split(
            fns,
            labels,
            IDs,
            stratify=labels,
            shuffle=True,
            test_size=0.3,
            random_state=42
            )
        nrrdss = [data_train, data_test]
        labelss = [label_train, label_test]
        IDss = [ID_train, ID_test]
        fn_arrs = ['train_arr.npy', 'test_arr.npy']
        fn_dfs = ['train_img_df.csv', 'test_img_df.csv'] 
        ## creat numpy array for image slices
        for nrrds, labels, IDs, fn_arr, fn_df in zip(nrrdss, labelss, IDss, fn_arrs, fn_dfs):
            img_dataset(
                pro_data_dir=pro_data_dir,
                run_type='tune',
                nrrds=nrrds,
                IDs=IDs,
                labels=labels,
                fn_arr_1ch=None,
                fn_arr_3ch=fn_arr,
                fn_df=fn_df,
                slice_range=slice_range,
                input_channel=3,
                norm_type=norm_type,
                )
        print('train and test datasets created!')   
    
    ## use entire exval data to test model
    elif split == False:
        nrrds = fns
        labels = labels
        IDs = IDs    
        img_dataset(
            pro_data_dir=pro_data_dir,
            run_type='tune',
            nrrds=nrrds,
            IDs=IDs,
            labels=labels,
            fn_arr_1ch=None,
            fn_arr_3ch='train_arr.npy',
            fn_df='train_img_df.csv',
            slice_range=slice_range,
            input_channel=3,
            norm_type=norm_type,
            )
        print('total patient:', len(IDs))
        print('exval datasets created!')

