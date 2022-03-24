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
from utils.nrrd_reg import nrrd_reg_rigid_ref
from utils.crop_image import crop_image
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image
import SimpleITK as sitk

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
        #print(nrrd)
        nrrd = sitk.ReadImage(nrrd, sitk.sitkFloat32)
        img_arr = sitk.GetArrayFromImage(nrrd)
        #print(img_arr.shape)
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
        #print(img_df[0:100])
        img_df.to_csv(os.path.join(pro_data_dir, fn_df))
        #print('data size:', img_df.shape[0])


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
            )
