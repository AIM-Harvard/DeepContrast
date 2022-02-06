from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import SimpleITK as sitk

#------------------------------------------------------------------------------------
# find last conv layer
#-----------------------------------------------------------------------------------
def find_target_layer(cnn_model):

    # find the final conv layer by looping layers in reverse order
    for layer in reversed(cnn_model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

#----------------------------------------------------------------------------------
# calculate gradient class actiavtion map
#----------------------------------------------------------------------------------
def compute_heatmap(cnn_model, image, pred_index, last_conv_layer):

    """
    construct our gradient model by supplying (1) the inputs
    to our pre-trained model, (2) the output of the (presumably)
    final 4D layer in the network, and (3) the output of the
    softmax activations from the model
    """
    gradModel = Model(
        inputs=[cnn_model.inputs],
        outputs=[cnn_model.get_layer(last_conv_layer).output, cnn_model.output]
        )

    # record operations for automatic differentiation
    with tf.GradientTape() as tape:
        """
        cast the image tensor to a float-32 data type, pass the
        image through the gradient model, and grab the loss
        associated with the specific class index
        """
        #print(pred_index)
        inputs = tf.cast(image, tf.float32)
        #print(image.shape)
        last_conv_layer_output, preds = gradModel(inputs)
        #print(preds)
        #print(preds.shape)
        # class_channel = preds[:, pred_index]
        class_channel = preds
    
    ## use automatic differentiation to compute the gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    """
    This is a vector where each entry is the mean intensity of the gradient
    over a specific feature map channel
    """
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    """
    We multiply each channel in the feature map array
    by "how important this channel is" with regard to the top predicted class
    then sum all the channels to obtain the heatmap class activation
    """
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

#------------------------------------------------------------------------------------
# save gradcam heat map
#-----------------------------------------------------------------------------------
def save_gradcam(run_type, img_back, heatmap, val_gradcam_dir, test_gradcam_dir, 
                 exval2_gradcam_dir, alpha, img_id):
    
#    print('heatmap:', heatmap.shape)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap('jet')
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # resize heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap0 = jet_heatmap.resize(re_size)
    jet_heatmap1 = keras.preprocessing.image.img_to_array(jet_heatmap0)
#    print('jet_heatmap:', jet_heatmap1.shape)

    # resize background CT image
    img = img_back.reshape((192, 192, 3))
    img = keras.preprocessing.image.array_to_img(img)
    ## resize if resolution of raw image too low
    #img0 = img.resize(re_size)
    img1 = keras.preprocessing.image.img_to_array(img)
#    print('img shape:', img1.shape)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap1 * alpha + img1
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    if run_type == 'val':
        save_dir = val_gradcam_dir
    elif run_type == 'test':
        save_dir = test_gradcam_dir
    elif run_type == 'exval2':
        save_dir = exval2_gradcam_dir

    fn1 = str(img_id) + '_' + str(conv_n) + '_' + 'gradcam.png'
    fn2 = str(img_id) + '_' + str(conv_n) + '_' + 'heatmap.png'
    fn3 = str(img_id) + '_' + str(conv_n) + '_' + 'heatmap_raw.png'
    fn4 = str(img_id) + '_' + 'CT.png'
    superimposed_img.save(os.path.join(save_dir, fn1))
#    jet_heatmap0.save(os.path.join(save_dir, fn2))
#    jet_heatmap.save(os.path.join(save_dir, fn3))
#    img0.save(os.path.join(save_dir, fn4))

#---------------------------------------------------------------------------------
# get background image
#---------------------------------------------------------------------------------
def get_background(img_id, slice_range, PMH_reg_dir, CHUM_reg_dir, CHUS_reg_dir, 
                   MDACC_reg_dir):
    
    pat_id = img_id.split('_')[0]
    if pat_id[:-3] == 'PMH':
        reg_dir = PMH_reg_dir
    elif pat_id[:-3] == 'CHUM':
        reg_dir = CHUM_reg_dir
    elif pat_id[:-3] == 'CHUS':
        reg_dir = CHUS_reg_dir
    elif pat_id[:-3] == 'MDACC':
        reg_dir = MDACC_reg_dir
    elif pat_id[:4] == 'rtog':
        reg_dir = rtog_reg_dir
        pat_id = img_id.split('_s')[0]

    nrrd_id = str(pat_id) + '.nrrd'
    data_dir = os.path.join(reg_dir, nrrd_id)
    ### get image slice and save them as numpy array
    nrrd = sitk.ReadImage(data_dir, sitk.sitkFloat32)
    img_arr = sitk.GetArrayFromImage(nrrd)
    print(img_arr.shape)
    data = img_arr[slice_range, :, :]
    #slice_n = img_id.split('_')[1][6:]
    slice_n = img_id.split('slice0')[1]
    slice_n = int(slice_n)
    print(slice_n)
    arr = data[slice_n, :, :]
    arr = np.clip(arr, a_min=-200, a_max=200)
    MAX, MIN = arr.max(), arr.min()
    arr = (arr - MIN) / (MAX - MIN)
    #print(arr.shape)
    arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = arr.reshape((1, 192, 192, 3))
    #print(arr.shape)
    img_back = arr
    #np.save(os.path.join(pro_data_dir, fn_arr_3ch), img_arr)
    
    return img_back

#---------------------------------------------------------------------------------
# get data
#---------------------------------------------------------------------------------
def gradcam(run_type, input_channel, img_IDs, conv_list, val_dir, test_dir, 
            exval_dir, model_dir, saved_model, data_pro_dir, pro_data_dir,
            run_model):
    
    ## load model and find conv layers
    cnn_model = load_model(os.path.join(model_dir, saved_model))
    # model.summary()

    ### load train data based on input channels
    if run_type == 'val':
        if input_channel == 1:
            fn = 'val_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'val_arr_3ch.npy'
        data = np.load(os.path.join(data_pro_dir, fn))
        df = pd.read_csv(os.path.join(pro_data_dir, 'val_img_pred.csv'))
        save_dir = val_dir
    elif run_type == 'test':
        if input_channel == 1:
            fn = 'test_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'test_arr_3ch.npy'
        data = np.load(os.path.join(data_pro_dir, fn))
        df = pd.read_csv(os.path.join(test_dir, 'df_test_pred.csv'))
        save_dir = test_dir
    elif run_type == 'exval2':
        if input_channel == 1:
            fn = 'exval_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'rtog_arr.npy'
        data = np.load(os.path.join(pro_data_dir, fn))
        df = pd.read_csv(os.path.join(pro_data_dir, 'rtog_img_pred.csv'))
        save_dir = exval2_dir
    print("successfully load data!")
    print(img_IDs)
    print(df[0:10])
    ## load data for gradcam    
    img_inds = df[df['fn'].isin(img_IDs)].index.tolist()
    print(img_inds)
    if img_inds == []:
        print("list is empty. Choose other slices.")
    else:
        for i, img_id in zip(img_inds, img_IDs):
            print('image ID:', img_id)
            print('index:', i)
            image = data[i, :, :, :]
            image = image.reshape((1, 192, 192, 3))
            label = df['label'][i]
            pred_index = df['y_pred_class'][i]
            y_pred = df['y_pred'][i]
            ## get background CT image
            img_back = get_background(
                img_id=img_id, 
                slice_range=slice_range, 
                PMH_reg_dir=PMH_reg_dir, 
                CHUM_reg_dir=CHUM_reg_dir, 
                CHUS_reg_dir=CHUS_reg_dir,
                MDACC_reg_dir=MDACC_reg_dir
                )
            if run_model == 'ResNet101V2':
                for conv_n in conv_list:
                    if conv_n == 'conv2':
                        last_conv_layer = 'conv2_block3_1_conv'
                    elif conv_n == 'conv3':
                        last_conv_layer = 'conv3_block4_1_conv'
                    elif conv_n == 'conv4':
                        last_conv_layer = 'conv4_block6_1_conv'
                    elif conv_n == 'conv5':
                        last_conv_layer = 'conv5_block3_out'
            elif run_model == 'EfficientNetB4':
                last_conv_layer = 'top_conv'
                #last_conv_layer = 'top_activation'
            ## compute heatnap
            heatmap = compute_heatmap(
                cnn_model=cnn_model,
                image=image,
                pred_index=pred_index,
                last_conv_layer=last_conv_layer
                )
            ## save heatmap
            save_gradcam(
                run_type=run_type,
                img_back=img_back,
                heatmap=heatmap,
                val_gradcam_dir=val_gradcam_dir,
                test_gradcam_dir=test_gradcam_dir,
                exval2_gradcam_dir=exval2_gradcam_dir,
                alpha=alpha,
                img_id=img_id
                )

            print('label:', label)
            print('ID:', img_id)
            print('y_pred:', y_pred)
            print('pred class:', pred_index)
            #print('conv layer:', conv_n)
#---------------------------------------------------------------------------------
# get data
#---------------------------------------------------------------------------------
if __name__ == '__main__':
    
    train_img_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/train_img_dir'
    val_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/val'
    test_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/test'
    exval_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval'
    exval2_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval2'
    val_gradcam_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/val/gradcam'
    test_gradcam_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/test/gradcam'
    exval_gradcam_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval/gradcam'
    exval2_gradcam_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval2/gradcam'
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    model_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/model'
    data_pro_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro'
    CHUM_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/CHUM_data_reg'
    CHUS_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/CHUS_data_reg'
    PMH_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/PMH_data_reg'
    MDACC_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/MDACC_data_reg'
    rtog_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/ahmed_data/rtog-0617_reg'
    input_channel = 3
    re_size = (192, 192)
    crop = True
    alpha = 0.9
   
    run_type = 'exval2'
    run_model = 'EfficientNetB4'
    if run_type in ['exval', 'exval2']:
        slice_range = range(50, 120)
        slice_ids = ["{0:03}".format(i) for i in range(70)]
        #saved_model = 'FineTuned_model_2021_08_02_17_22_10'
        saved_model = 'Tuned_EfficientNetB4_2021_08_27_20_26_55'
    elif run_type in ['val', 'test']:
        slice_range = range(17, 83)
        slice_ids = ["{0:03}".format(i) for i in range(66)]
        #saved_model = 'ResNet_2021_07_18_06_28_40'
        saved_model = 'EffNet_2021_08_24_09_57_13'
    print(run_type) 
    print(slice_range)
    print(slice_ids)    
    show_network = False
    conv_n = 'conv5'
    conv_list = ['conv2', 'conv3', 'conv4', 'conv5']
    
    ## image ID
    pat_id = 'PMH423'
    pat_ids = ['rtog_0617-438343']
    pat_ids = ['PMH574', 'PMH146', 'PMH135']
    pat_ids = ['PMH433', 'PMH312', 'PMH234', 'PMH281', 'PMH511', 'PMH405']
    pat_ids = ['PMH465', 'PMH287', 'PMH308', 'PMH276', 'PMH595', 'PMH467']
    pat_ids = ['rtog_0617-349454', 'rtog_0617-438343', 'rtog_0617-292370', 'rtog_0617-349454']
    img_IDs = []
    for pat_id in pat_ids:
        for slice_id in slice_ids:
            img_id = pat_id + '_' + 'slice' + str(slice_id)
            img_IDs.append(img_id)

    gradcam(
        run_type=run_type,
        input_channel=input_channel, 
        img_IDs=img_IDs, 
        conv_list=conv_list, 
        val_dir=val_dir,
        test_dir=test_dir,
        exval_dir=exval_dir,
        model_dir=model_dir,
        saved_model=saved_model,
        data_pro_dir=data_pro_dir,
        pro_data_dir=pro_data_dir,
        run_model=run_model
        )
        


            
#    if last_conv_layer is None:
#        last_conv_layer = find_target_layer(
#            model=model,
#            saved_model=saved_model
#            )
#    print(last_conv_layer)
#
#    if show_network == True:
#        for idx in range(len(model.layers)):
#            print(model.get_layer(index = idx).name)

#    # compute the guided gradients
#    castConvOutputs = tf.cast(convOutputs > 0, "float32")
#    castGrads = tf.cast(grads > 0, "float32")
#    guidedGrads = castConvOutputs * castGrads * grads
#    # the convolution and guided gradients have a batch dimension
#    # (which we don't need) so let's grab the volume itself and
#    # discard the batch
#    convOutputs = convOutputs[0]
#    guidedGrads = guidedGrads[0]
#
#    # compute the average of the gradient values, and using them
#    # as weights, compute the ponderation of the filters with
#    # respect to the weights
#    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
#    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
#
#    # grab the spatial dimensions of the input image and resize
#    # the output class activation map to match the input image
#    # dimensions
## (w, h) = (image.shape[2], image.shape[1])
## heatmap = cv2.resize(cam.numpy(), (w, h))
#    heatmap = cv2.resize(heatmap.numpy(), (64, 64))
#    # normalize the heatmap such that all values lie in the range
##    # [0, 1], scale the resulting values to the range [0, 255],
##    # and then convert to an unsigned 8-bit integer
#    numer = heatmap - np.min(heatmap)
#    eps = 1e-8
#    denom = (heatmap.max() - heatmap.min()) + eps
#    heatmap = numer / denom
#    heatmap = (heatmap * 255).astype("uint8")
#    colormap=cv2.COLORMAP_VIRIDIS
#    heatmap = cv2.applyColorMap(heatmap, colormap)
#    print('heatmap shape:', heatmap.shape)
##    img = image[:, :, :, 0]
##    print('img shape:', img.shape)
#    img = image.reshape((64, 64, 3))
#    print(img.shape)
#    output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
#   
#   
#    return heatmap, output
