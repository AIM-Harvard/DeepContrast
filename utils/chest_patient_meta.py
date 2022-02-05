"""
  ----------------------------------------
  get patient and CT metadata for chest CT
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.8.8
  ----------------------------------------
  After the data (CT-mask pair, or just CT) is processed by the first script,
  export downsampled versions to be used for heart-localisation purposes.
  During this downsampling step, resample and crop/pad images - log all the
  information needed for upsampling (and thus obtain a rough segmentation that
  will be used for the localisation).
  
"""


import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

#----------------------------------------------------------------------------------------
# external val dataset using lung CT
#----------------------------------------------------------------------------------------
def chest_metadata(harvard_rt_dir, data_exclude, pro_data_dir, data_pro_dir, split):
  
    """

    Get patient and scan metadata for chest CT

    @params:
    data_sitk   - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    new_spacing - required : desired spacing (equal for all the axes), in mm, of the output data
    method      - required : SimpleITK interpolation method (e.g., sitk.sitkLinear)

    FIXME: change this into something like downsample_sitk (also, data_sitk to img_sitk for homog.)
    (as this function is especially used for downsampling, right?)

    """

    #----------------------------
    ## rotg_0617 test dataset
    #-----------------------------
    df_pat = pd.read_csv(os.path.join(pro_data_dir, 'rtog_pat_df.csv'))
    df = pd.read_csv(os.path.join(pro_data_dir, 'rtog_final_curation.csv'))
    df['gender'].replace([1], 0, inplace=True)
    df['gender'].replace([2], 1, inplace=True)

    IDs = []
    ages = []
    genders = []
    stages = []
    thks = []    
    histologys = []
    sizes = []
    spacings = []   
    print(df['patid'])
    print(df_pat['ID']) 
    for patid, age, gender, stage, thk, histology, size, spacing in zip(
            df['patid'], 
            df['age'], 
            df['gender'], 
            df['ajcc_stage_grp'],
            df['spacing_Z'],
            df['histology'],
            df['size_X'],
            df['spacing_X'],
            ):
        if patid in df_pat['ID'].to_list():
            IDs.append(patid)
            ages.append(age)
            genders.append(gender)
            stages.append(stage)
            thks.append(thk)
            histologys.append(histology)
            sizes.append(size)
            spacings.append(spacing)
    
    ## patient meta - test set
    df_test = pd.DataFrame({
        'ID': IDs,
        'gender': genders,
        'age': ages,
        'stage': stages,
        'histology': histologys
        })
    print('df_test:', df_test.shape[0])
    
    ## CT scan meta data
    df_scan1 = pd.DataFrame({
        'ID': IDs,
        'thk': thks,
        'size': sizes,
        'spacing': spacings
        })
    print('df_scan1:', df_scan1.shape[0])

    #---------------------------------------
    ## harvard-rt train and val dataset
    #----------------------------------------
    df1 = pd.read_csv(os.path.join(data_pro_dir, 'harvard_rt_meta.csv'))
    df1.dropna(subset=['ctdose_contrast', 'top_coder_id'], how='any', inplace=True)
    df2 = pd.read_csv(os.path.join(data_pro_dir, 'harvard_rt.csv'))
     
    ## all scan ID to list
    IDs = []
    list_fn = [fn for fn in sorted(glob.glob(harvard_rt_dir + '/*nrrd'))]
    for fn in list_fn:
        ID = fn.split('/')[-1].split('.')[0].strip()
        IDs.append(ID)
    print('IDs:', len(IDs))
    print('top coder ID:', df1['top_coder_id'].shape[0]) 

    #------------------------------
    # meta file 1 - harvard_rt_meta
    #------------------------------
    genders = []
    scanners = []
    kvps = []
    thks = []
    tstages = []
    nstages = []
    mstages = []
    stages = []
    labels = []
    for top_coder_id, label, gender, scanner, \
        kvp, thk, tstage, nstage, mstage, stage in zip(
            df1['top_coder_id'],
            df1['ctdose_contrast'], 
            df1['gender'],
            df1['scanner_type'],
            df1['kvp_value'],
            df1['slice_thickness'],
            df1['clin_tstage'],
            df1['clin_nstage'],
            df1['clin_mstage'],
            df1['clin_stage']
            ):
        tc_id = top_coder_id.split('_')[2].strip()
        if tc_id in IDs:
            labels.append(label)
            genders.append(gender)
            scanners.append(scanner)
            kvps.append(kvp)
            thks.append(thk)
            tstages.append(tstage)
            nstages.append(nstage)
            mstages.append(mstage)
            stages.append(stage)
    
    #-------------------------
    # meta file 2 - harvard_rt
    #-------------------------
    ages = []
    histologys = []
    sizes = []
    spacings = []    
    for topcoder_id, age, histology, size, spacing in zip(
            df2['topcoder_id'],
            df2['age'],
            df2['histology'],
            df2['raw_size_x'],
            df2['raw_spacing_x'],
            ):
        if topcoder_id in IDs:
            ages.append(age)
            histologys.append(histology)
            sizes.append(size)
            spacings.append(spacing)

    ## delete excluded scans and repeated scans
    if data_exclude != None:
        df_exclude = df[df['ID'].isin(data_exclude)]
        print('exclude scans:', df_exclude)
        df.drop(df[df['ID'].isin(test_exclude)].index, inplace=True)
        print('total scans:', df.shape[0])
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    #print(df[0:50])
    
    #---------------------------------------------------
    # split dataset for fine-tuning model and test model
    #---------------------------------------------------
    if split == True:
        ID1, ID2, gender1, gender2, age1, age2, tstage1, tstage2, nstage1, \
        nstage2, mstage1, mstage2, stage1, stage2, histo1, histo2 = train_test_split(
            IDs,
            genders,
            ages,
            tstages,
            nstages,
            mstages,
            stages,
            histologys,
            stratify=labels,
            shuffle=True,
            test_size=0.2,
            random_state=42
            )
    
    ## patient meta - train df
    df_train = pd.DataFrame({
        'ID': ID1,
        'gender': gender1,
        'age': age1,
        'stage': stage1,
        'histology': histo1,
        })
    #df.to_csv(os.path.join(pro_data_dir, 'exval_pat_df.csv'))
    print('train set:', df_train.shape[0])
    
    ## patient meta - val df
    df_val = pd.DataFrame({
        'ID': ID2,
        'gender': gender2,
        'age': age2,
        'stage': stage2,
        'histology': histo2,
        })
    print('val set:', df_val.shape[0])
   
    ## patient meta - train + val - test
    df_tot = pd.concat([df_train, df_val, df_test])

    ## print patient meta
    dfs = [df_train, df_val, df_test, df_tot]
    datasets = ['train', 'val', 'test', 'all']
    
    for df, dataset in zip(dfs, datasets):
        print('\n')
        print('----------------------------')
        print(dataset)
        print('----------------------------')
        print('patient number:', df.shape[0])
        print('median age:', df['age'].median().round(3))
        print('age max:', df['age'].max().round(3))
        print('age min:', df['age'].min().round(3))
        print('\n')
        print(df['gender'].value_counts())
        print(df['gender'].value_counts(normalize=True).round(3))
        print('\n')
        print(df['stage'].value_counts())
        print(df['stage'].value_counts(normalize=True).round(3))
        print('\n')
        print(df['histology'].value_counts())
        print(df['histology'].value_counts(normalize=True).round(3))    
    
    #-----------------------------------------
    ## print scan meta data
    #-----------------------------------------
    ## CT scan meta data
    df_scan2 = pd.DataFrame({
        'ID': IDs,
        'thk': thks,
        'size': sizes,
        'spacing': spacings
        })
    #print('scanner:', scanners)
    #print('kvp:', kvps)
    ## scan parameters
    df_scan3 = pd.DataFrame({
        'scanner': scanners,
        'kvp': kvps,
        })
    
    ## print scan metadata
    df_scan = pd.concat([df_scan1, df_scan2])
    df = df_scan
    print('\n')
    print('-------------------')
    print('thickness and size')
    print('-------------------')
    print('print scan metadata:')
    print('patient number:', df.shape[0])
    print(df['thk'].value_counts())
    print(df['thk'].value_counts(normalize=True).round(3))
    print(df['size'].value_counts())
    print(df['size'].value_counts(normalize=True).round(3))
    ## slice thickness
    print('\n')
    print('-------------------')
    print('slice thickness')
    print('-------------------')
    print('thk mean:', df['thk'].mean().round(3))
    print('thk median:', df['thk'].median())
    print('thk mode:', df['thk'].mode())
    print('thk std:', df['thk'].std().round(3))
    print('thk min:', df['thk'].min())
    print('thk max:', df['thk'].max())
    ##  voxel spacing
    print('\n')
    print('-------------------')
    print('spacing info')
    print('-------------------')
    print('spacing mean:', df['spacing'].mean().round(3))
    print('spacing median:', df['spacing'].median())
    print('spacing mode:', df['spacing'].mode())
    print('spacing std:', df['spacing'].std().round(3))
    print('spacing min:', df['spacing'].min())
    print('spacing max:', df['spacing'].max())
    
    df = df_scan3
    print('\n')
    print('-------------------')
    print('scanner info')
    print('-------------------')
    print('patient number:', df.shape[0])
    ## scanner type
    print(df['scanner'].value_counts())
    print(df['scanner'].value_counts(normalize=True).round(3))
    ## tued voltage (kvp)
    print('\n')
    print('-------------------')
    print('KVP')
    print('-------------------')
    print('kvp mean:', df['kvp'].mean().round(3))
    print('kvp median:', df['kvp'].median())
    print('kvp mode:', df['kvp'].mode())
    print('kvp std:', df['kvp'].std().round(3))
    print('kvp min:', df['kvp'].min())
    print('kvp max:', df['kvp'].max())

#-----------------------------------------------------------------------------------
# run funtions
#-----------------------------------------------------------------------------------
if __name__ == '__main__':
    
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    data_pro_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro'
    harvard_rt_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/NSCLC_data_reg'

    chest_metadata(
        harvard_rt_dir=harvard_rt_dir, 
        data_exclude=None, 
        pro_data_dir=pro_data_dir, 
        data_pro_dir=data_pro_dir,
        split=True,
        )


