import glob
import shutil
import os
import pandas as pd
import nrrd
import re
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from time import gmtime, strftime
from datetime import datetime
import timeit
from utils.respacing import respacing
from utils.nrrd_reg import nrrd_reg_rigid_reffrom utils.crop_image import crop_image
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image
import SimpleITK as sitk


def preprocess_data(data_dir, pre_data_dir, new_spacing, data_exclude=None,
                    crop_shape=[192, 192, 10], interp_type='linear'):

    """
    Preprocess data including: respacing, registration, cropping;

    Args:
        data_dir {path} -- path to CT data;
        out_dir {path} -- path to result outputs;
    Keyword args:
        new_spacing {tuple} -- respacing size;
        return_type {str} -- image data format after preprocessing, default: 'nrrd';
        data_exclude {str} -- exclude patient data due to data issue, default: None;
        crop_shape {np.array} -- numpy array size afer cropping;
        interp_type {str} -- interpolation type for respacing, default: 'linear';
    Return:
        save nrrd image data;
    """

    reg_temp_img = os.path.join(data_dir, 'HN001.nrrd')
    fns = [fn for fn in sorted(glob.glob(data_dir + '/*nrrd'))]
    ## patient ID
    IDs = []
    for fn in fns:
        ID = fn.split('/')[-1].split('.')[0].strip()
        IDs.append(ID)
    ## PMH dataframe
    df = pd.DataFrame({'ID': IDs, 'file': fns})
    for fn, ID in zip(df['file'], df['ID']):
        print(ID)
        ## respacing      
        img_nrrd = respacing(
            nrrd_dir=fn,
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type='nrrd',
            save_dir=None)
        ## registration
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_nrrd,
            fixed_img_dir=reg_temp_img,
            patient_id=ID,
            save_dir=None)
        ## crop image from (500, 500, 116) to (180, 180, 60)
        img_crop = crop_image(
            nrrd_file=img_reg,
            patient_id=ID,
            crop_shape=crop_shape,
            return_type='nrrd',
            save_dir=data_reg_diri)


def get_pat_dataset(data_dir, data_reg_dir, label_dir, pro_data_dir):

    """
    create dataframe to contain data path, patient ID and label on the 
    patient level;

    Args:
        label_dir {path} -- path for label csv file;
        label_file {csv} -- csv file contain lable info;
        cohort {str} -- patient cohort name (PMH, CHUM, MDACC, CHUS);
        MDACC_data_dir {patyh} -- path to MDACC patient data;
    Return:
        panda dataframe for patient data;

    """
 
    ## labels
    df_label = pd.read_csv(os.path.join(label_dir, label_file))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    ## data
    fns = [fn for fn in sorted(glob.glob(data_reg_dir + '/*nrrd'))]
    ## patient ID
    IDs = []
    for fn in fns:
        ID = fn.split('/')[-1].split('.')[0].strip()
        IDs.append(ID)

    ## create dataframe
    print('ID:', len(IDs))
    print('file:', len(fns))
    print('label:', len(labels))
    df = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    data = df['file']
    label = df['label']
    ID = df['ID']
    data_train_, data_test, label_train_, label_test, ID_train_, ID_test = train_test_split(
        data,
        label,
        ID,
        stratify=label,
        test_size=0.3,
        random_state=42)
    data_train, data_val, label_train, label_val, ID_train, ID_val = train_test_split(
        data_train_,
        label_train_,
        ID_train_,
        stratify=label,
        test_size=0.3,
        random_state=42)
    ## save train, val, test df on patient level
    train_pat_df = pd.DataFrame({'ID': ID_train, 'file': data_train, 'label': label_train})
    val_pat_df = pd.DataFrame({'ID': ID_val, 'file': data_val, 'label': label_val})
    test_pat_df = pd.DataFrame({'ID': ID_test, 'file': data_test, 'label': label_test})
    train_pat_df.to_csv(os.path.join(pro_data_dir, 'train_pat_df.csv'))
    val_pat_df.to_csv(os.path.join(pro_data_dir, 'val_pat_df.csv'))
    test_pat_df.to_csv(os.path.join(pro_data_dir, 'test_pat_df.csv'))
    ## save data, label and ID as list
    data_tot = [data_train, data_val, data_test]
    label_tot = [label_train, label_val, label_test]
    ID_tot = [ID_train, ID_val, ID_test]

    return data_tot, label_tot, ID_tot


 def img_dataset(pro_data_dir, run_type, nrrds, IDs, labels, fn_arr_1ch, fn_arr_3ch, fn_df,
                slice_range, input_channel=3, norm_type='np_clip'):

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

    # get image slice and save them as numpy array
    count = 0
    slice_numbers = []
    list_fn = []
    arr = np.empty([0, 192, 192])

    for nrrd, patient_id in zip(nrrds, IDs):
        count += 1
        print(count)
        nrrd = sitk.ReadImage(nrrd, sitk.sitkFloat32)
        img_arr = sitk.GetArrayFromImage(nrrd)
        #data = img_arr[30:78, :, :]
        #data = img_arr[17:83, :, :]
        data = img_arr[slice_range, :, :]
        ### clear signals lower than -1024
        data[data <= -1024] = -1024
        ### strip skull, skull UHI = ~700
        data[data > 700] = 0
        ### normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
        if norm_type == 'np_interp':
            data = np.interp(data, [-200, 200], [0, 1])
        elif norm_type == 'np_clip':
            data = np.clip(data, a_min=-200, a_max=200)
            MAX, MIN = data.max(), data.min()
            data = (data - MIN) / (MAX - MIN)
        ## stack all image arrays to one array for CNN input
        arr = np.concatenate([arr, data], 0)

        ### create patient ID and slice index for img
        slice_numbers.append(data.shape[0])
        for i in range(data.shape[0]):
            img = data[i, :, :]
            fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
            list_fn.append(fn)

    ### covert 1 channel input to 3 channel inputs for CNN
    if input_channel == 1:
        img_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_1ch), img_arr)
    elif input_channel == 3:
        img_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img_arr = np.transpose(img_arr, (1, 2, 3, 0))
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_3ch), img_arr)
        #fn = os.path.join(pro_data_dir, 'exval_arr_3ch.h5')
        #h5f = h5py.File(fn, 'w')
        #h5f.create_dataset('dataset_exval_arr_3ch', data=img_arr)
    ### generate labels for CT slices
    if run_type == 'pred':
        ### makeing dataframe containing img dir and labels
        img_df = pd.DataFrame({'fn': list_fn})
        img_df.to_csv(os.path.join(pro_data_dir, fn_df))
        print('data size:', img_df.shape[0])
    else:
        list_label = []
        list_img = []
        for label, slice_number in zip(labels, slice_numbers):
            list_1 = [label] * slice_number
            list_label.extend(list_1)
        ### makeing dataframe containing img dir and labels
        img_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
        pd.options.display.max_columns = 100
        pd.set_option('display.max_rows', 500)
        print(img_df[0:100])
        img_df.to_csv(os.path.join(pro_data_dir, fn_df))
        print('data size:', img_df.shape[0])

        
def get_img_dataset(pro_data_dir, run_type, data_tot, ID_tot, label_tot, slice_range):

    """
    Get np arrays for stacked images slices, labels and IDs for train, val, test dataset;

    Args:
        run_type {str} -- train, val, test, external val, pred;
        pro_data_dir {path} -- path to processed data;
        data_tot {list} -- list of data paths: ['data_train', 'data_val', 'data_test'];
        ID_tot {list} -- list of image IDs: ['ID_train', 'ID_val', 'ID_test'];
        label_tot {list} -- list of image labels: ['label_train', 'label_val', 'label_test'];
        slice_range {np.array} -- image slice range in z direction for cropping;

    """

    fns_arr_1ch = ['train_arr_1ch.npy', 'val_arr_1ch.npy', 'test_arr_1ch.npy']
    fns_arr_3ch = ['train_arr_3ch.npy', 'val_arr_3ch.npy', 'test_arr_3ch.npy']
    fns_df = ['train_img_df.csv', 'val_img_df.csv', 'test_img_df.csv']

    for nrrds, IDs, labels, fn_arr_1ch, fn_arr_3ch, fn_df in zip(
        data_tot, ID_tot, label_tot, fns_arr_1ch, fns_arr_3ch, fns_df):

        img_dataset(
            pro_data_dir=pro_data_dir,
            run_type=run_type,
            nrrds=nrrds,
            IDs=IDs,
            labels=labels,
            fn_arr_1ch=fn_arr_1ch,
            fn_arr_3ch=fn_arr_3ch,
            fn_df=fn_df,
            slice_range=slice_range,
            input_channel=input_channel,
            norm_type=norm_type,
            )


def dataset(data_dir, pre_data_dir, pro_data_dir, label_dir, data_exclude, new_spacing, crop_shape, 
            interp_type, run_type, slice_range):

    preprocess_data(
        data_dir=data_dir,
        data_reg_dir=pre_data_dir,
        new_spacing=new_spacing,
        data_exclude=data_exclude,
        crop_shape=crop_shape
        interp_type=interp_type)

    data_tot, label_tot, ID_tot = get_pat_dataset(
        data_dir=data_dir,
        data_reg_dir=pre_data_dir,
        label_dir=label_dir,
        pro_data_dir=pro_data_dir)

    get_img_dataset(
        pro_data_dir=pro_data_dir,
        run_type=run_type,
        data_tot=data_tot,
        ID_tot=ID_tot,
        label_tot=label_tot,
        slice_range=slice_range)


