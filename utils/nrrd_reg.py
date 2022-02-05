import sys, os, glob
import SimpleITK as sitk
#import pydicom
import numpy as np


def nrrd_reg_rigid_ref(img_nrrd, fixed_img_dir, patient_id, save_dir):
     
    fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
    moving_img = img_nrrd
#    moving_img = sitk.ReadImage(img_nrrd, sitk.sitkUInt32)
    #moving_img = sitk.ReadImage(input_path, sitk.sitkFloat32)
    
    transform = sitk.CenteredTransformInitializer(
        fixed_img, 
        moving_img, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    
    # multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10
        )
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)
    final_transform = registration_method.Execute(fixed_img, moving_img)                               
    moving_img_resampled = sitk.Resample(
        moving_img, 
        fixed_img, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_img.GetPixelID()
        )
    img_reg = moving_img_resampled
    
    if save_dir != None:   
        nrrd_fn = str(patient_id) + '.nrrd' 
        sitk.WriteImage(img_red, os.path.join(save_dir, nrrd_fn))

    return img_reg
    #return fixed_img, moving_img, final_transform
