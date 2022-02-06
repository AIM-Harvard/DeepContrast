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



def preprocess_data(data_dir, out_dir, new_spacing=(1, 1, 3), data_exclude=None, 
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
 
    CHUM_data_dir = os.path.join(data_dir, '0_image_raw_CHUM')
    CHUS_data_dir = os.path.join(data_dir, '0_image_raw_CHUS')
    PMH_data_dir = os.path.join(data_dir, '0_image_raw_PMH')
    MDACC_data_dir = os.path.join(data_dir, '0_image_raw_MDACC')
    CHUM_reg_dir = os.path.join(out_dir, 'data/CHUM_data_reg')
    CHUS_reg_dir = os.path.join(out_dir, 'data/CHUS_data_reg')
    PMH_reg_dir = os.path.join(out_dir, 'data/PMH_data_reg')
    MDACC_reg_dir = os.path.join(out_dir, 'data/MDACC_data_reg')
    
    if not os.path.exists(CHUM_reg_dir):
        os.mkdir(CHUM_reg_dir)
    if not os.path.exists(CHUS_reg_dir):
        os.mkdir(CHUS_reg_dir)
    if not os.path.exists(PMH_reg_dir):
        os.mkdir(PMH_reg_dir)
    if not os.path.exists(MDACC_reg_dir):
        os.mkdir(MDACC_reg_dir)

    reg_temp_img = os.path.join(PMH_reg_dir, 'PMH050.nrrd')

    # get PMH data
    #------------------
    fns = [fn for fn in sorted(glob.glob(PMH_data_dir + '/*nrrd'))]
    ## PMH patient ID
    IDs = []
    for fn in fns:
        ID = 'PMH' + fn.split('/')[-1].split('-')[1][2:5].strip()
        IDs.append(ID)
    ## PMH dataframe
    df_PMH = pd.DataFrame({'ID': IDs, 'file': fns})
    pd.options.display.max_colwidth = 100
    #print(df_pmh)
    file = df_PMH['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('PMH data:', len(IDs))
    
    # get CHUM data
    #-----------------
    fns = []
    for fn in sorted(glob.glob(CHUM_data_dir + '/*nrrd')):
        scan_ = fn.split('/')[-1].split('_')[2].strip()
        if scan_ == 'CT-SIM':
            fns.append(fn)
        else:
            continue
    ## CHUM patient ID
    IDs = []
    for fn in fns:
        ID = 'CHUM' + fn.split('/')[-1].split('_')[1].split('-')[2].strip()
        IDs.append(ID)
    ## CHUM dataframe
    df_CHUM = pd.DataFrame({'ID': IDs, 'file': fns})
    #print(df_chum)
    pd.options.display.max_colwidth = 100
    file = df_CHUM['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('CHUM data:', len(IDs))

    # get CHUS data
    #---------------
    fns = []
    for fn in sorted(glob.glob(CHUS_data_dir + '/*nrrd')):
        scan = fn.split('/')[-1].split('_')[2].strip()
        if scan == 'CT-SIMPET':
            fns.append(fn)
        else:
            continue
    ## CHUS patient ID
    IDs = []
    for fn in fns:
        ID = 'CHUS' + fn.split('/')[-1].split('_')[1].split('-')[2].strip()
        IDs.append(ID)
    ## CHUS dataframe
    df_CHUS = pd.DataFrame({'ID': IDs, 'file': fns})
    pd.options.display.max_colwidth = 100
    #print(df_chus)
    file = df_CHUS['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('CHUS data:', len(IDs))

    # get MDACC dataset
    #------------------
    fns = [fn for fn in sorted(glob.glob(MDACC_data_dir + '/*nrrd'))]
    ## MDACC patient ID
    IDs = []
    for fn in fns:
        ID = 'MDACC' + fn.split('/')[-1].split('-')[2][1:4].strip()
        IDs.append(ID)
    ## MDACC dataframe
    df_MDACC = pd.DataFrame({'ID': IDs, 'file': fns})
    df_MDACC.drop_duplicates(subset=['ID'], keep='last', inplace=True)
    print('MDACC data:', df_MDACC.shape[0])

    # combine dataset for train-val split
    #-------------------------------------
    df = pd.concat([df_PMH, df_CHUM, df_CHUS, df_MDACC], ignore_index=True)
    print('total patients:', df.shape[0])
    #print(df[700:])
    ## exclude data with certain conditions
    if data_exclude != None:
        df_exclude = df[df['ID'].isin(data_exclude)]
        print(df_exclude)
        df.drop(df[df['ID'].isin(data_exclude)].index, inplace=True)
        print(df.shape[0])
    
    for fn, ID in zip(df['file'], df['ID']):
   
        print(ID)
    
        ## set up save dir
        if ID[:-3] == 'PMH':
            save_dir = PMH_reg_dir
        elif ID[:-3] == 'CHUM':
            save_dir = CHUM_reg_dir
        elif ID[:-3] == 'CHUS':
            save_dir = CHUS_reg_dir
        elif ID[:-3] == 'MDACC':
            save_dir = MDACC_reg_dir
        
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
            save_dir=save_dir
            ) 
