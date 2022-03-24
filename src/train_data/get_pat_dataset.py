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


def get_pat_dataset(data_dir, pre_data_dir, label_dir, label_file, pro_data_dir):

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
    labels = df_label['Contrast'].to_list()
    ## data
    fns = [fn for fn in sorted(glob.glob(pre_data_dir + '/*nrrd'))]
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
        stratify=label_train_,
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
