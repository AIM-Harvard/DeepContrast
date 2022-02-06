import os
import pandas as pd
import numpy as np
import glob


pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'

## actual data from df
train_data = pd.read_csv(os.path.join(pro_data_dir, 'train_pat_df.csv'))
val_data = pd.read_csv(os.path.join(pro_data_dir, 'val_pat_df.csv'))
test_data = pd.read_csv(os.path.join(pro_data_dir, 'test_pat_df.csv'))
data = pd.concat([train_data, val_data, test_data])

## clinical meta data    
df = pd.read_csv(os.path.join(pro_data_dir, 'clinical_meta_data.csv'))
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
print('mismatch ID:', ids)
print('mismatch label:', labels)        
print('mismatch label:', contrasts)
print('mismatch number:', len(contrasts))

print('total patient:', df['contrast'].shape[0])
print(df['contrast'].value_counts())
print(df['contrast'].value_counts(normalize=True))

