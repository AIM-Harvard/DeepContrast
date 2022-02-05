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
from respacing import respacing
from nrrd_reg import nrrd_reg_rigid_ref
from crop_image import crop_image



def save_prepro_scan(PMH_data_dir, CHUM_data_dir, CHUS_data_dir, MDACC_data_dir,
                     PMH_reg_dir, CHUM_reg_dir, CHUS_reg_dir, MDACC_reg_dir, fixed_img_dir,
                     interp_type, new_spacing, return_type, data_exclude, crop_shape):
   
    for fn, ID in zip(fns, IDs):
        
        print(ID)

        ## set up save dir
        if ID[:-3] == 'PMH':
            save_dir = PMH_reg_dir
            file_dir = os.path.join(PMH_data_dir, fn)
        elif ID[:-3] == 'CHUM':
            save_dir = CHUM_reg_dir
            file_dir = os.path.join(CHUM_data_dir, fn)
        elif ID[:-3] == 'CHUS':
            save_dir = CHUS_reg_dir
            file_dir = os.path.join(CHUS_data_dir, fn)
        elif ID[:-3] == 'MDACC':
            save_dir = MDACC_reg_dir
            file_dir = os.path.join(MDACC_data_dir, fn)

        ## respacing      
        img_nrrd = respacing(
            nrrd_dir=file_dir,
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type=return_type,
            save_dir=None
            )

        ## registration
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_nrrd,
            fixed_img_dir=fixed_img_dir,
            patient_id=ID,
            save_dir=None
            )

        ## crop image from (500, 500, 116) to (180, 180, 60)
        img_crop = crop_image(
            nrrd_file=img_reg,
            patient_id=ID,
            crop_shape=crop_shape,
            return_type='nrrd',
            save_dir=save_dir
            )

if __name__ == '__main__':

    
    fns = [
           'mdacc_HNSCC-01-0028_CT-SIM-09-06-1999-_raw_raw_raw_xx.nrrd',
           'mdacc_HNSCC-01-0168_CT-SIM-11-19-2001-_raw_raw_raw_xx.nrrd',  
           'mdacc_HNSCC-01-0181_CT-SIM-07-16-2002-_raw_raw_raw_xx.nrrd' 
           ]
    IDs = ['MDACC028', 'MDACC168', 'MDACC181']

    val_save_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/val'
    test_save_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test'
    MDACC_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_mdacc'
    PMH_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_pmh'
    CHUM_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_chum'
    CHUS_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_chus'
    CHUM_reg_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/data/CHUM_data_reg'
    CHUS_reg_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/data/CHUS_data_reg'
    PMH_reg_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/data/PMH_data_reg'
    MDACC_reg_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/data/MDACC_data_reg'
    data_pro_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/data_pro'
    fixed_img_dir = os.path.join(data_pro_dir, 'PMH050.nrrd')
    input_channel = 3
    crop = True
    n_img = 20
    save_img = False
    data_exclude = None
    thr_prob = 0.5
    run_type = 'test'
    norm_type = 'np_clip'
    new_spacing = [1, 1, 3]
    crop_shape = [192, 192, 100]
    return_type = 'nrrd'
    interp_type = 'linear'
    
    save_prepro_scan(
        PMH_data_dir=PMH_data_dir, 
        CHUM_data_dir=CHUM_data_dir, 
        CHUS_data_dir=CHUS_data_dir, 
        MDACC_data_dir=MDACC_data_dir,
        PMH_reg_dir=PMH_reg_dir, 
        CHUM_reg_dir=CHUM_reg_dir, 
        CHUS_reg_dir=CHUS_reg_dir, 
        MDACC_reg_dir=MDACC_reg_dir, 
        fixed_img_dir=fixed_img_dir,
        interp_type=interp_type, 
        new_spacing=new_spacing, 
        return_type=return_type, 
        data_exclude=data_exclude, 
        crop_shape=crop_shape
        )
    print("successfully save prepro scan!")
