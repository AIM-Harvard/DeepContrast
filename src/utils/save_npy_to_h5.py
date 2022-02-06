import numpy as np
import h5py

file_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro/exval_arr_3ch1.npy'
data = np.load(file_dir)

with h5py.File('exval_arr_3ch1.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=data_to_write)

print("H5 created.")
