import pandas as pd
import os


#data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/ahmed_data'
#df = pd.read_csv(os.path.join(data_dir, 'rtog-0617_pat.csv'))
#df.drop(df.index[[343]], inplace=True)
#df.to_csv(os.path.join(data_dir, 'rtog-0617_pat.csv'))


data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
df = pd.read_csv(os.path.join(data_dir, 'rtog_img_df.csv'))
df.drop(df.index[range(24010, 24080)], axis=0, inplace=True)
df.to_csv(os.path.join(data_dir, 'rtog_img_df.csv'))
