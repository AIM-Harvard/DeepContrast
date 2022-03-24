
vimport glob
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
from get_data.get_img_dataset import img_dataset



def data_prepro(body_part, data_dir, new_spacing=[1, 1, 3], 
                input_channel=3, norm_type='np_clip'):
   
    """
    data preprocrssing: respacing, registration, crop
    
    Arguments:
        crop_shape {np.array} -- array shape for cropping image.
        fixed_img_dir {str} -- dir for registered template iamge. 
        data_dir {str} -- data dir.
        slice_range {np.array} -- slice range to extract axial slices of scans.

    Keyword arguments:
        input_channel {int} -- input channel 1 or 3.
        new_spacing {np.array} -- respacing size, default [1, 1, 3].
        norm_type {'str'} -- normalization methods for image, 'np_clip' or 'np_interp'
      
    return:
        df_img {pd.df} -- dataframe with image ID and patient ID.
        img_arr {np.array}  -- stacked numpy array from all slices of all scans.
    
        
    """
    

    if body_part == 'HeadNeck':
        crop_shape = [192, 192, 100]
        slice_range = range(17, 83)
        data_dir = os.path.join(data_dir, 'HeadNeck')
    elif body_part == 'Chest':
        crop_shape = [192, 192, 140]
        slice_range = range(50, 120)
        data_dir = os.path.join(data_dir, 'Chest')
    
    # choose first scan as registration template
    reg_template = sorted(glob.glob(data_dir + '/*nrrd'))[0]

    # registration, respacing, cropping   
    img_ids = []
    pat_ids = []
    slice_numbers = []
    arr = np.empty([0, 192, 192])
    for fn in sorted(glob.glob(data_dir + '/*nrrd')):
        pat_id = fn.split('/')[-1].split('.')[0].strip()
        print(pat_id)
        ## respacing      
        img_nrrd = respacing(
            nrrd_dir=fn,
            interp_type='linear',
            new_spacing=new_spacing,
            patient_id=pat_id,
            return_type='nrrd',
            save_dir=None
            )
        ## registration
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_nrrd,
            fixed_img_dir=reg_template,
            patient_id=pat_id,
            save_dir=None
            )
        ## crop image from (500, 500, 116) to (180, 180, 60)
        img_crop = crop_image(
            nrrd_file=img_reg,
            patient_id=pat_id,
            crop_shape=crop_shape,
            return_type='npy',
            save_dir=None
            )
        
        ## choose slice range to cover body part
        if slice_range == None:
            data = img_crop
        else:
            data = img_crop[slice_range, :, :]
        ## clear signals lower than -1024
        data[data <= -1024] = -1024
        ## strip skull, skull UHI = ~700
        data[data > 700] = 0
        ## normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
        if norm_type == 'np_interp':
            data = np.interp(data, [-200, 200], [0, 1])
        elif norm_type == 'np_clip':
            data = np.clip(data, a_min=-200, a_max=200)
            MAX, MIN = data.max(), data.min()
            data = (data - MIN) / (MAX - MIN)
        ## stack all image arrays to one array for CNN input
        arr = np.concatenate([arr, data], 0)

        ## create image ID and slice index for img
        slice_numbers.append(data.shape[0])
        for i in range(data.shape[0]):
            img = data[i, :, :]
            img_id = pat_id + '_' + 'slice%s'%(f'{i:03d}')
            img_ids.append(img_id)
            pat_ids.append(pat_id)

    # generate patient and slice ID
    df_img = pd.DataFrame({'pat_id': pat_ids, 'img_id': img_ids})
    #print('data size:\n', df_img)

    # covert 1 channel input to 3 channel inputs for CNN
    if input_channel == 1:
        img_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        #print('img_arr shape:', img_arr.shape)
        #np.save(os.path.join(pro_data_dir, fn_arr_1ch), img_arr)
    elif input_channel == 3:
        img_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img_arr = np.transpose(img_arr, (1, 2, 3, 0))

    return df_img, img_arr 

