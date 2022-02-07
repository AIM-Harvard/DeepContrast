import os
import itertools
import operator
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import nrrd
from PIL import Image


#--------------------------------------------------------------------------------------
# crop image
#-------------------------------------------------------------------------------------
def crop_image(nrrd_file, patient_id, crop_shape, return_type, save_dir):
    
    ## load stik and arr
    img_arr = sitk.GetArrayFromImage(nrrd_file)
    ## Return top 25 rows of 3D volume, centered in x-y space / start at anterior (y=0)?
#    img_arr = np.transpose(img_arr, (2, 1, 0))
#    print("image_arr shape: ", img_arr.shape)
    c, y, x = img_arr.shape
#    x, y, c = image_arr.shape
#    print('c:', c)
#    print('y:', y)
#    print('x:', x)
    
    ## Get center of mass to center the crop in Y plane
    mask_arr = np.copy(img_arr) 
    mask_arr[mask_arr > -500] = 1
    mask_arr[mask_arr <= -500] = 0
    mask_arr[mask_arr >= -500] = 1 
    #print("mask_arr min and max:", np.amin(mask_arr), np.amax(mask_arr))
    centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
    cpoint = c - crop_shape[2]//2
    #print("cpoint, ", cpoint)
    centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])   
    #print("center of mass: ", centermass)
    startx = int(centermass[0] - crop_shape[0]//2)
    starty = int(centermass[1] - crop_shape[1]//2)      
    #startx = x//2 - crop_shape[0]//2       
    #starty = y//2 - crop_shape[1]//2
    startz = int(c - crop_shape[2])
    #print("start X, Y, Z: ", startx, starty, startz)
    
    ## crop image using crop shape
    if startz < 0:
        img_arr = np.pad(
            img_arr,
            ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 
            'constant', 
            constant_values=-1024
            )
        img_crop_arr = img_arr[
            0:crop_shape[2], 
            starty:starty + crop_shape[1], 
            startx:startx + crop_shape[0]
            ]
    else:
        img_crop_arr = img_arr[
#           0:crop_shape[2],
            startz:startz + crop_shape[2], 
            starty:starty + crop_shape[1], 
            startx:startx + crop_shape[0]
            ]
    if img_crop_arr.shape[0] < crop_shape[2]:
        #print('initial cropped image shape too small:', img_arr.shape)
        #print(crop_shape[2], img_crop_arr.shape[0])
        img_crop_arr = np.pad(
            img_crop_arr,
            ((int(crop_shape[2] - img_crop_arr.shape[0]), 0), (0, 0), (0, 0)),
            'constant',
            constant_values=-1024
            )
        #print("padded size: ", img_crop_arr.shape)
    #print(img_crop_arr.shape)    
    ## get nrrd from numpy array
    img_crop_nrrd = sitk.GetImageFromArray(img_crop_arr)
    img_crop_nrrd.SetSpacing(nrrd_file.GetSpacing())
    img_crop_nrrd.SetOrigin(nrrd_file.GetOrigin())

    if save_dir != None:
        fn = str(patient_id) + '.nrrd'
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(save_dir, fn))
        writer.SetUseCompression(True)
        writer.Execute(img_crop_nrrd)

    if return_type == 'nrrd':
        return img_crop_nrrd

    elif return_type == 'npy':
        return img_crop_arr
 
#-----------------------------------------------------------------------
# run codes to test
#-----------------------------------------------------------------------
if __name__ == '__main__':

#    output_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/output'
    output_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test'
    file_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_PMH'
    test_file = 'PMH_OPC-00050_CT-SIM_raw_raw_raw_xx.nrrd'
    return_type = 'sitk'
    crop_shape = [192, 192, 110]

    train_file = os.path.join(file_dir, test_file)

    img_crop = crop_image(
        nrrd_file=train_file,
        crop_shape=crop_shape,
        return_type=return_type,
        output_dir=output_dir
        )
    print('crop arr shape:', img_crop)
#    arr_crop = image_arr_crop[0, :, :]
#    img_dir = os.path.join(output_dir, 'arr_crop.jpg')
#    plt.imsave(img_dir, arr_crop, cmap='gray')
    print('successfully save image!!!')
    

