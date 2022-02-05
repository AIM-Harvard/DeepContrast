import pandas as pd
import numpy as np
import os
import glob


#-------------------------------------------------------------------------
# create patient df with ID, label and dir
#-------------------------------------------------------------------------
def get_pat_df(pro_data_dir, reg_data_dir, label_file, fn_pat_df):   
    
    ## create df for dir, ID and labels on patient level
    df_label = pd.read_csv(os.path.join(pro_data_dir, label_file))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    labels = df_label['Contrast'].to_list()
    fns = [fn for fn in sorted(glob.glob(reg_data_dir + '/*nrrd'))]
    IDs = []
    for fn in fns:
        ID = fn.split('/')[-1].split('_')[1].split('.')[0].strip()
        IDs.append(ID)    
    pat_ids = []
    labels = []
    for pat_id, pat_label in zip(df_label['Patient ID'], df_label['Contrast']):
        if pat_id in IDs:
            pat_id = 'rtog' + '_' + str(pat_id)
            pat_ids.append(pat_id)
            labels.append(pat_label)
    print("ID:", len(pat_ids))
    print("dir:", len(fns))
    print("label:", len(labels))
    print('contrast scan in ex val:', labels.count(1))
    print('non-contrast scan in ex val:', labels.count(0))
    df = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    df.to_csv(os.path.join(pro_data_dir, fn_pat_df))
    print('total scan:', df.shape[0])

#-------------------------------------------------------------------------
# create img df with ID, label
#-------------------------------------------------------------------------
def get_img_df(pro_data_dir, fn_pat_df, fn_img_df, slice_range):
    
    pat_df = pd.read_csv(os.path.join(pro_data_dir, fn_pat_df))
    ## img ID
    slice_number = len(slice_range)
    img_ids = []
    for pat_id in pat_df['ID']:
        for i in range(slice_number):
            img_id = 'rtog' + '_' + pat_id + '_' + 'slice%s'%(f'{i:03d}')
            img_ids.append(img_id)
    #print(img_ids)
    print(len(img_ids))
    ## img label
    pat_label = pat_df['label'].to_list()
    img_label = []
    for label in pat_label:
        list2 = [label] * slice_number
        img_label.extend(list2)
    #print(img_label) 
    print(len(img_label))
    ### makeing dataframe containing img IDs and labels
    df = pd.DataFrame({'fn': img_ids, 'label': img_label})
    print(df[0:10])
    df.to_csv(os.path.join(pro_data_dir, fn_img_df))
    print('total img:', df.shape[0])

#-------------------------------------------------------------------------
# create patient df with ID, label and dir
#-------------------------------------------------------------------------    
if __name__ == '__main__':
    
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    slice_range = range(50, 120)
    reg_data_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/ahmed_data/rtog-0617_reg'
    label_file = 'label_RTOG0617.csv'
    fn_pat_df = 'rtog_pat_df.csv'
    fn_img_df = 'rtog_img_df.csv'

    get_pat_df(
        pro_data_dir=pro_data_dir, 
        reg_data_dir=reg_data_dir, 
        label_file=label_file, 
        fn_pat_df=fn_pat_df
        )

    get_img_df(
        pro_data_dir=pro_data_dir,
        fn_pat_df=fn_pat_df,
        fn_img_df=fn_img_df,
        slice_range=slice_range
        )




