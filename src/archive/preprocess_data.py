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



def preprocess_data(data_dir, data_reg_dir, new_spacing, data_exclude=None, 
                    crop_shape=[192, 192, 10], interp_type='linear'):
    
    """
    Preprocess data including: respacing, registration, cropping;

    Arguments:
        data_dir {path} -- path to CT data;
        out_dir {path} -- path to result outputs;
    
    Keyword arguments:
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
            save_dir=data_reg_dir
            ) 



