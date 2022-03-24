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




def pat_df(label_dir, label_file, cohort, data_reg_dir, MDACC_data_dir):

    """
    create dataframe to contain data path, patient ID and label on the 
    patient level;

    Arguments:
        label_dir {path} -- path for label csv file;
        label_file {csv} -- csv file contain lable info;
        cohort {str} -- patient cohort name (PMH, CHUM, MDACC, CHUS);
        MDACC_data_dir {patyh} -- path to MDACC patient data;

    Return:
        panda dataframe for patient data;

    """

    ## labels
    labels = []
    df_label = pd.read_csv(os.path.join(label_dir, label_file))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    if cohort == 'CHUM':
        for file_ID, label in zip(df_label['File ID'], df_label['Contrast']):
            scan = file_ID.split('_')[2].strip()
            if scan == 'CT-SIM':
                labels.append(label)
            elif scan == 'CT-PET':
                continue
    elif cohort == 'CHUS':
        labels = df_label['Contrast'].to_list()
    elif cohort == 'PMH':
        labels = df_label['Contrast'].to_list()
    elif cohort == 'MDACC':
        fns = [fn for fn in sorted(glob.glob(MDACC_data_dir + '/*nrrd'))]
        IDs = []
        for fn in fns:
            ID = 'MDACC' + fn.split('/')[-1].split('-')[2][1:4].strip()
            IDs.append(ID)
        labels = df_label['Contrast'].to_list()
        print('MDACC label:', len(labels))
        print('MDACC ID:', len(IDs))
        ## check if labels and data are matched
        for fn in fns:
            fn = fn.split('/')[-1]
            if fn not in df_label['File ID'].to_list():
                print(fn)
        ## make df and delete duplicate patient scans
        df = pd.DataFrame({'ID': IDs, 'labels': labels})
        df.drop_duplicates(subset=['ID'], keep='last', inplace=True)
        labels = df['labels'].to_list()
        #print('MDACC label:', len(labels))
    
    ## data
    fns = [fn for fn in sorted(glob.glob(data_reg_dir + '/*nrrd'))]
    
    ## patient ID
    IDs = []
    for fn in fns:
        ID = fn.split('/')[-1].split('.')[0].strip()
        IDs.append(ID)
    ## check id and labels
    if cohort == 'MDACC':
        list1 = list(set(IDs) - set(df['ID'].to_list()))
        print(list1)
    ## create dataframe
    print('cohort:', cohort)
    print('ID:', len(IDs))
    print('file:', len(fns))
    print('label:', len(labels))
    df = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    
    return df



def get_pat_dataset(data_dir, out_dir, proj_dir):

    """
    get data path, patient ID and label for all the cohorts;

    Arguments:
        data_dir {path} -- path to the CT data;
        lab_drive_dir {path} -- path to outputs;
        proj_dir {path} -- path to processed data;
        CHUM_label_csv {csv} -- label file for CHUM cohort;
        CHUS_label_csv {csv} -- label file for CHUS cohort;
        PMH_label_csv {csv} -- label file for PMH cohort;
        MDACC_label_csv {csv} -- label file for MDACC cohort;

    Return:
        lists for patient data, labels and IDs;

    """
    
    MDACC_data_dir = os.path.join(data_dir, '0_image_raw_MDACC')
    CHUM_reg_dir = os.path.join(lab_drive_dir, 'data/CHUM_data_reg')
    CHUS_reg_dir = os.path.join(lab_drive_dir, 'data/CHUS_data_reg')
    PMH_reg_dir = os.path.join(lab_drive_dir, 'data/PMH_data_reg')
    MDACC_reg_dir = os.path.join(lab_drive_dir, 'data/MDACC_data_reg')
    label_dir = os.path.join(lab_drive_dir, 'data_pro')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    
    cohorts = ['CHUM', 'CHUS', 'PMH', 'MDACC']
    label_files = ['label_CHUM.csv', 'label_CHUS.csv', 'label_PMH.csv', 'label_MDACC.csv']
    data_reg_dirs = [CHUM_reg_dir, CHUS_reg_dir, PMH_reg_dir, MDACC_reg_dir]
    df_tot = []
    for cohort, label_file, data_reg_dir in zip(cohorts, label_files, data_reg_dirs):
        df = pat_df(
            label_dir=label_dir,
            label_file=label_file,
            cohort=cohort,
            data_reg_dir=data_reg_dir,
            MDACC_data_dir=MDACC_data_dir
            )
        df_tot.append(df) 

    ## get df for different cohorts
    df_CHUM = df_tot[0]
    df_CHUS = df_tot[1]
    df_PMH = df_tot[2]
    df_MDACC = df_tot[3]

    ## train-val split
    df = pd.concat([df_PMH, df_CHUM, df_CHUS], ignore_index=True)
    data = df['file']
    label = df['label']
    ID = df['ID']
    data_train, data_val, label_train, label_val, ID_train, ID_val = train_test_split(
        data,
        label,
        ID,
        stratify=label,
        test_size=0.3,
        random_state=42
        )

    ## test patient data
    data_test = df_MDACC['file']
    label_test = df_MDACC['label']
    ID_test = df_MDACC['ID']
  
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

