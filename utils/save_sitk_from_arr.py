#--------------------------------------------------------------------------------------------
# save image as itk
#-------------------------------------------------------------------------------------------
def save_sitk_from_arr(img_sitk, new_arr, resize, save_dir):

    """
    When resize == True: Used for saving predictions where padding needs to be added to increase the size 
    of the prediction and match that of input to model. This function matches the size of the array in 
    image_sitk_obj with the size of pred_arr, and saves it. This is done equally on all sides as the 
    input to model and model output have different dims to allow for shift data augmentation.
    When resize == False: the image_sitk_obj is only used as a reference for spacing and origin. The numpy 
    array is not resized.
    image_sitk_obj: sitk object of input to model
    pred_arr: returned prediction from model - should be squeezed.
    NOTE: image_arr.shape will always be equal or larger than pred_arr.shape, but never smaller given that
    we are always cropping in data.py
    """

    if resize == True:
        # get array from sitk object
        img_arr = sitk.GetArrayFromImage(img_sitk)
        # change pred_arr.shape to match image_arr.shape
        # getting amount of padding needed on each side
        z_diff = int((img_arr.shape[0] - new_arr.shape[0]) / 2)
        y_diff = int((img_arr.shape[1] - new_arr.shape[1]) / 2)
        x_diff = int((img_arr.shape[2] - new_arr.shape[2]) / 2)
        # pad, defaults to 0
        new_arr = np.pad(new_arr, ((z_diff, z_diff), (y_diff, y_diff), (x_diff, x_diff)), 'constant')
        assert img_arr.shape == new_arr.shape, "returned array shape does not match your requested shape."

    # save sitk obj
    new_sitk = sitk.GetImageFromArray(new_arr)
    new_sitk.SetSpacing(img_sitk.GetSpacing())
    new_sitk.SetOrigin(img_sitk.GetOrigin())

    if output_dir != None:
#        fn = "{}_{}_image_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id)
        fn = 'test_stik.nrrd'
        img_dir = os.path.join(output_dir, fn)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(img_dir)
        writer.SetUseCompression(True)
        writer.Execute(new_sitk)

    return new_sitk
