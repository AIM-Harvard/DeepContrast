import pandas as pd
import os

proj_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/ahmed_data/results'
df_pred = pd.read_csv(os.path.join(proj_dir, 'rtog-0617_pat_pred.csv'))
df_id = pd.read_csv(os.path.join(proj_dir, 'check_list.csv'))
list_id = df_id['patient_id'].to_list()

IDs = []
preds = []
for ID, pred in zip(df_pred['ID'], df_pred['predictions']):
    if ID.split('_')[1] in list_id:
        IDs.append(ID)
        preds.append(pred)

df_check = pd.DataFrame({'ID': IDs, 'pred': preds})
df_check.to_csv(os.path.join(proj_dir, 'rtog_check.csv'), index=False)
