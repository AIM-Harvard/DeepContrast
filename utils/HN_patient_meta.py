import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from plot_cm import plot_cm


def pat_meta_info(pro_data_dir):
  
    """

    Get patient and scan metadata for chest CT

    @params:
    data_sitk   - required : SimpleITK image, resulting from sitk.ImageFileReader().Execute()
    new_spacing - required : desired spacing (equal for all the axes), in mm, of the output data
    method      - required : SimpleITK interpolation method (e.g., sitk.sitkLinear)

    FIXME: change this into something like downsample_sitk (also, data_sitk to img_sitk for homog.)
    (as this function is especially used for downsampling, right?)

    """


    #-----------------------------------------
    # CT scan artifacts
    #-----------------------------------------
    df = pd.read_csv(os.path.join(pro_data_dir, 'ContrastAnnotation_HN.csv'))
    df.drop_duplicates(subset=['Patient ID'], keep='last', inplace=True)
    df.dropna(subset=['Artifact-OP'], inplace=True)
    print('annotation data with no duplicates:', df.shape[0])

    train_data = pd.read_csv(os.path.join(pro_data_dir, 'train_pat_df.csv'))
    val_data = pd.read_csv(os.path.join(pro_data_dir, 'val_pat_df.csv'))
    test_data = pd.read_csv(os.path.join(pro_data_dir, 'test_pat_df.csv'))
    tot_data = pd.concat([train_data, val_data, test_data])
    datas = [tot_data, train_data, val_data, test_data]
    names = ['tot', 'train', 'val', 'test']
    
    ## artifacts for train, val, test and tot dataset
    for name, data in zip(names, datas):
        pat_ids = []
        artifacts = []
        for patientid, artifact, note in zip(df['Patient ID'], df['Artifact-OP'], df['Notes']):
            ## use consistent ID
            if patientid[3:7] == 'CHUM':
                pat_id = 'CHUM' + patientid[-3:]
            elif patientid[3:7] == 'CHUS':
                pat_id = 'CHUS' + patientid[-3:]
            elif patientid[:3] == 'OPC':
                pat_id = 'PMH' + patientid[-3:]
            elif patientid[:5] == 'HNSCC':
                pat_id = 'MDACC' + patientid[-3:]
            ## find very severe artifacts
            if note == 'really bad artifact':
                artifact = 'very bad'
            else:
                artifact = artifact
            ## append artifacts
            if pat_id in data['ID'].to_list():
                pat_ids.append(pat_id)
                artifacts.append(artifact)
          
        df_af = pd.DataFrame({'ID': pat_ids, 'Artifact-OP': artifacts})
        print('----------------------------')
        print(name)
        print('----------------------------')
        print('data size:', df_af.shape[0])
        print('data with artifact:', df_af.loc[df_af['Artifact-OP'].isin(['Bad', 'Yes', 'Minimal'])].shape[0])
        print(df_af['Artifact-OP'].value_counts())
        print(df_af['Artifact-OP'].value_counts(normalize=True).round(3))
    
    #-----------------------------------------
    # clean and group metadata
    #-----------------------------------------
    df = pd.read_csv(os.path.join(pro_data_dir, 'clinical_meta_data.csv'))
    print('\nmeta data size:', df.shape[0])
    df.drop_duplicates(subset=['patientid'], keep='last', inplace=True)
    print('meta data with no duplicates:', df.shape[0])

    ## combine HPV info from tow cols
    hpvs = []
    df['hpv'] = df.iloc[:, 8].astype(str) + df.iloc[:, 9].astype(str)
    for hpv in df['hpv']:
        if hpv in ['nannan', 'Unknownnan', 'Nnan', 'Not testednan', 'no tissuenan']:
            hpv = 'unknown'
        elif hpv in ['  positivenan', 'Pnan', '+nan', 'nanpositive', 'Positivenan', 
                     'Positive -Strongnan', 'Positive -focalnan']:
            hpv = 'positive'
        elif hpv in ['  Negativenan', 'Negativenan', '-nan', 'nannegative']:
            hpv = 'negative'
        hpvs.append(hpv)
    df['hpv'] = hpvs
    
    ## overall stage
    stages = []
    for stage in df['ajccstage']:
        if stage in ['I', 'Stade I']:
            stage = 'I'
        elif stage in ['II', 'Stade II', 'StageII']:
            stage = 'II'
        elif stage in ['III', 'Stade III', 'Stage III']:
            stage = 'III'
        elif stage in ['IVA', 'IV', 'IVB', 'Stade IVA', 'Stage IV', 'Stade IVB']:
            stage = 'IV'
        stages.append(stage)
    df['ajccstage'] = stages

    ## primary cancer sites
    sites = []
    for site in df['diseasesite']:
        if site in ['Oropharynx']:
            site = site
        elif site in ['Larynx', 'Hypopharynx', 'Nasopharynx']:
            site = 'Larynx/Hypopharynx/Nasopharynx'
        elif site in ['Oral cavity']:
            site = site
        else:
            site = 'Unknown/Other'
        sites.append(site)
    df['diseasesite'] = sites
    
    ## sex
    df['gender'].replace(['F'], 'Female', inplace=True)
    df['gender'].replace(['M'], 'Male', inplace=True)
    
    #-----------------------------------------
    # patient meta data
    #-----------------------------------------
    ## actual patient data with images
    train_data = pd.read_csv(os.path.join(pro_data_dir, 'train_pat_df.csv'))
    val_data = pd.read_csv(os.path.join(pro_data_dir, 'val_pat_df.csv'))
    test_data = pd.read_csv(os.path.join(pro_data_dir, 'test_pat_df.csv'))
    print('train data:', train_data.shape[0])
    print('val data:', val_data.shape[0])
    print('test data:', test_data.shape[0])
    
    ## print contrast info in train, val, test sets
    datas = [train_data, val_data, test_data]
    names = ['train', 'val', 'test']
    for data, name in zip(datas, names):
        print('\n')
        print('----------------------------')
        print(name)
        print('----------------------------')
        print(data['label'].value_counts())
        print(data['label'].value_counts(normalize=True).round(3))

    ## find patient metadata
    datas = [train_data, val_data, test_data]
    metas = []
    for data in datas:
        ids = []
        genders = [] 
        ages = []
        tcats = []
        stages = []
        sites = []
        ncats = []
        hpvs = []
        ## find meta info
        for patientid, gender, age, t_cat, ajccstage, site, n_cat, hpv in zip(
            df['patientid'], df['gender'], df['ageatdiag'], df['t-category'], 
            df['ajccstage'], df['diseasesite'], df['n-category'], df['hpv']):
            ## 4 datasets
            if patientid[3:7] == 'CHUM':
                pat_id = 'CHUM' + patientid[-3:]
            elif patientid[3:7] == 'CHUS':
                pat_id = 'CHUS' + patientid[-3:]
            elif patientid[:3] == 'OPC':
                pat_id = 'PMH' + patientid[-3:]
            elif patientid[:5] == 'HNSCC':
                pat_id = 'MDACC' + patientid[-3:]
            if pat_id in data['ID'].to_list():
                #print(pat_id)
                ids.append(patientid) 
                genders.append(gender) 
                ages.append(age)
                tcats.append(t_cat)
                stages.append(ajccstage)
                sites.append(site)
                ncats.append(n_cat)
                hpvs.append(hpv)
        ## create new df for train, val, test meta info
        meta = pd.DataFrame(
                            {'id': ids,
                             'gender': genders,
                             'age': ages,
                             't_stage': tcats,
                             'stage': stages,
                             'site': sites,
                             'n_stage': ncats,
                             'hpv': hpvs}
                             ) 
        metas.append(meta)
    ## concat 3 datasets to 1 big dataset
    all_meta = pd.concat([metas[0], metas[1], metas[2]])
    metas.append(all_meta)   
    ## print meta info
    datasets = ['train', 'val', 'test', 'all']
    for df, dataset in zip(metas, datasets):
        print('\n')
        print('----------------------------')
        print(dataset)
        print('----------------------------')
        print('patient number:', df.shape[0])
        print('\n')
        print(df['gender'].value_counts())
        print(df['gender'].value_counts(normalize=True).round(3))
        print('\n')
        print(df['t_stage'].value_counts())
        print(df['t_stage'].value_counts(normalize=True).round(3))
        print('\n')
        print(df['stage'].value_counts())
        print(df['stage'].value_counts(normalize=True).round(3))
        print('\n')
        print(df['site'].value_counts())
        print(df['site'].value_counts(normalize=True).round(3))
        print('\n')
        print(df['n_stage'].value_counts())
        print(df['n_stage'].value_counts(normalize=True).round(3))
        print('\n')
        print(df['hpv'].value_counts())
        print(df['hpv'].value_counts(normalize=True).round(3))
        print('\n')
        print('mediam age:', df['age'].median())
        print('age max:', df['age'].max())
        print('age min:', df['age'].min())
        print('---------------------------------------------')

    #------------------------------------------------------------
    # CT meata data
    #------------------------------------------------------------
    df = pd.read_csv(os.path.join(pro_data_dir, 'clinical_meta_data.csv'))
    df.drop_duplicates(subset=['patientid'], keep='last', inplace=True)
    print(df.shape[0])
    print(all_meta.shape[0])
    df0 = df[~df['patientid'].isin(all_meta['id'].to_list())]
    df = df[~df['patientid'].isin(df0['patientid'].to_list())]  
    print('patient not in list:', df.shape[0])   
    
    ## combine CT scanner and model names
    IDs = []
    for manufacturer, model in zip(df['manufacturer'], df['manufacturermodelname']):
        ID = str(manufacturer) + ' ' + str(model)
        IDs.append(ID)
    df['ID'] = IDs
    #print(df['manufacturer'].value_counts())
    print('-------------------')
    print('CT scanner')
    print('-------------------')
    #print(df['manufacturermodelname'].value_counts())
    #print(df['manufacturermodelname'].value_counts(normalize=True).round(3))
    print(df['ID'].value_counts())
    print(df['ID'].value_counts(normalize=True).round(3))
    print(df.shape[0])

    ## KVP
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

    ## slice thickness
    print('\n')
    print('-------------------')
    print('slice thickness')
    print('-------------------')
    print('thk mean:', df['slicethickness'].mean().round(3))
    print('thk median:', df['slicethickness'].median())
    print('thk mode:', df['slicethickness'].mode())
    print('thk std:', df['slicethickness'].std().round(3))
    print('thk min:', df['slicethickness'].min())
    print('thk max:', df['slicethickness'].max())
    print(df['slicethickness'].value_counts())
    print(df['slicethickness'].shape[0])

    ## spatial resolution
    print('\n')
    print(df['rows'].value_counts())

    ## pixel spacing
    pixels = []
    for pixel in df['pixelspacing']:
        pixel = pixel.split("'")[1]
        pixel = float(pixel)
        pixels.append(pixel)
    df['pixel'] = pixels
    df['pixel'].round(3)
    print('\n')
    print('-------------------')
    print('pixel size')
    print('-------------------')
    print('pixel mean:', df['pixel'].mean().round(3))
    print('pixel median:', df['pixel'].median().round(3))
    print('pixel mode:', df['pixel'].mode().round(3))
    print('pixel std:', df['pixel'].std().round(3))
    print('pixel min:', df['pixel'].min().round(3))
    print('pixel max:', df['pixel'].max().round(3))

    data = pd.concat([train_data, val_data, test_data])
    
    #-----------------------------------------------------------------
    # contrast information from mata data
    #----------------------------------------------------------------
    df = pd.read_csv(os.path.join(pro_data_dir, 'clinical_meta_data.csv'))
    print('\n')
    print('-----------------------------------')
    print('contrast information from meta dta')
    print('-----------------------------------')
    print(df['contrastbolusagent'].value_counts())
    print(df['contrastbolusagent'].value_counts(normalize=True).round(3))
    list_contrast = set(df['contrastbolusagent'].to_list())
    print('contrast agents bolus number:', len(list_contrast))
    print(list_contrast)
    df['contrastbolusagent'] = df['contrastbolusagent'].fillna(2)

    pat_ids = []
    contrasts = []
    for patientid, contrast in zip(df['patientid'], df['contrastbolusagent']):
        if patientid[3:7] == 'CHUM':
            pat_id = 'CHUM' + patientid[-3:]
        elif patientid[3:7] == 'CHUS':
            pat_id = 'CHUS' + patientid[-3:]
        elif patientid[:3] == 'OPC':
            pat_id = 'PMH' + patientid[-3:]
        elif patientid[:5] == 'HNSCC':
            pat_id = 'MDACC' + patientid[-3:]
        if pat_id in data['ID'].to_list():
            pat_ids.append(pat_id)
            ## change contrast annotation in meta data
            if contrast in ['N', 'n', 'NO']:
                contrast = 0
            elif contrast == 2:
                contrast = contrast
            else:
                contrast = 1
            contrasts.append(contrast)
    df = pd.DataFrame({'ID': pat_ids, 'contrast': contrasts})
   
    ## match metadata annotations with clinical expert
    ids = []
    contrasts = []
    labels = []
    for ID, label in zip(data['ID'], data['label']):
        for pat_id, contrast in zip(df['ID'], df['contrast']):
            if pat_id == ID and contrast != 2 and contrast != label:
                ids.append(pat_id)
                contrasts.append(contrast)
                labels.append(label)
    print('\n')
    print('-----------------------------------')
    print('contrast information from meta dta')
    print('-----------------------------------')
    print('mismatch ID:', ids)
    print('mismatch label:', labels)
    print('mismatch label:', contrasts)
    print('mismatch number:', len(contrasts))
    print('total patient:', df['contrast'].shape[0])
    print(df['contrast'].value_counts())
    print(df['contrast'].value_counts(normalize=True).round(3))

    ## print contrast info in train, val, test sets
    datas = [train_data, val_data, test_data]
    names = ['train', 'val', 'test']
    conss = []
    for data, name in zip(datas, names):
        cons = []
        IDs = []
        labels = []
        for ID, label in zip(data['ID'], data['label']):
            for pat_id, con in zip(df['ID'], df['contrast']):
                if pat_id == ID:
                    cons.append(con)
                    labels.append(label)
                    IDs.append(pat_id)
            df_con = pd.DataFrame({'ID': IDs, 'label': labels, 'contrast': cons})
        conss.append(df_con)
    names = ['train', 'val', 'test']
    for name, con in zip(names, conss):        
        print('\n')
        print('----------------------------')
        print(name)
        print('----------------------------')
        print(con['contrast'].value_counts())
        print(con['contrast'].value_counts(normalize=True).round(3))
        #print(con['label'])

    #--------------------------------------------------------------------
    # calculate confusion matrix, accuracy and AUC for contrast metadata
    #--------------------------------------------------------------------
    for name, con in zip(['val', 'test'], [conss[1], conss[2]]):
        contrasts = []
        for contrast, label in zip(con['contrast'], con['label']):
            if contrast == 2 and label == 0:
                contrast = 1
            elif contrast == 2 and label == 1:
                contrast = 0
            else:
                contrast = contrast
            contrasts.append(contrast)
        con['contrast'] = contrasts
        cm = confusion_matrix(con['label'], con['contrast'])
        cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.around(cm_norm, 2)
        print('\n')
        print(name)
        print(cm_norm)
        print(cm)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        ACC = (TP + TN)/(TP + FP + FN + TN)
        TPR = TP/(TP + FN)
        TNR = TN/(TN + FP)
        AUC = (TPR + TNR)/2
        report = classification_report(con['label'], con['contrast'])
        print('AUC:', np.around(AUC[1], 3))
        print('ACC:', np.around(ACC[1], 3))
        print('report:', report)

        # plot confusion matrix
        save_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/metadata'
        for cm0, cm_type in zip([cm, cm_norm], ['raw', 'norm']):
            plot_cm(
                cm0=cm0,
                cm_type=cm_type,
                level=name,
                save_dir=save_dir
                )

if __name__ == '__main__':
    
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    
    pat_meta_info(pro_data_dir)


