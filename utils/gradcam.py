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

#---------------------------------------------------------------------------------
# get data
#---------------------------------------------------------------------------------
def data(input_channel, i, val_save_dir, test_save_dir):

    ### load train data based on input channels
    if run_type == 'val':
        if input_channel == 1:
            fn = 'val_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'val_arr_3ch.npy'
        data = np.load(os.path.join(pro_data_dir, fn))
        df = pd.read_csv(os.path.join(val_save_dir, 'val_pred_df.csv'))
    elif run_type == 'test':
        if input_channel == 1:
            fn = 'test_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'test_arr_3ch.npy'
        data = np.load(os.path.join(pro_data_dir, fn))
        df = pd.read_csv(os.path.join(test_save_dir, 'test_pred_df.csv'))
    elif run_type == 'exval':
        if input_channel == 1:
            fn = 'exval_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'exval_arr_3ch.npy'
        data = np.load(os.path.join(pro_data_dir, fn))
        df = pd.read_csv(os.path.join(exval_save_dir, 'exval_pred_df.csv'))

    ### load label
    y_true = df['label']
    y_pred_class = df['y_pred_class']
    y_pred = df['y_pred']  
    ID = df['fn']  
    ### find the ith image to show grad-cam map
    img = data[i, :, :, :]
    img = img.reshape((1, 192, 192, 3))
    label = y_true[i]
    pred_index = y_pred_class[i]
    y_pred = y_pred[i]
    ID = ID[i]

    return img, label, pred_index, y_pred, ID

#------------------------------------------------------------------------------------
# find last conv layer
#-----------------------------------------------------------------------------------
def find_target_layer(model, saved_model):

    # find the final conv layer by looping layers in reverse order
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

#----------------------------------------------------------------------------------
# calculate gradient class actiavtion map
#----------------------------------------------------------------------------------
def compute_heatmap(model, saved_model, image, pred_index, last_conv_layer):

    """
    construct our gradient model by supplying (1) the inputs
    to our pre-trained model, (2) the output of the (presumably)
    final 4D layer in the network, and (3) the output of the
    softmax activations from the model
    """
    gradModel = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer).output, model.output]
        )

    # record operations for automatic differentiation
    with tf.GradientTape() as tape:
        """
        cast the image tensor to a float-32 data type, pass the
        image through the gradient model, and grab the loss
        associated with the specific class index
        """
        print(pred_index)
        inputs = tf.cast(image, tf.float32)
        print(image.shape)
        last_conv_layer_output, preds = gradModel(inputs)
        print(preds)
        print(preds.shape)
    # class_channel = preds[:, pred_index]
        class_channel = preds
    # use automatic differentiation to compute the gradients
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
def save_gradcam(image, heatmap, val_gradcam_dir, test_gradcam_dir,  alpha, i):
    
#    print('heatmap:', heatmap.shape)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # resize heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap0 = jet_heatmap.resize(re_size)
    jet_heatmap1 = keras.preprocessing.image.img_to_array(jet_heatmap0)
#    print('jet_heatmap:', jet_heatmap1.shape)

    # resize background CT image
    img = image.reshape((192, 192, 3))
    img = keras.preprocessing.image.array_to_img(img)
    img0 = img.resize(re_size)
    img1 = keras.preprocessing.image.img_to_array(img0)
#    print('img shape:', img1.shape)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap1 * alpha + img1
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    if run_type == 'val':
        save_dir = val_gradcam_dir
    elif run_type == 'test':
        save_dir = test_gradcam_dir
    elif run_type == 'exval':
        save_dir = exval_gradcam_dir

    fn1 = str(conv_n) + '_' + str(i) + '_' + 'gradcam.png'
    fn2 = str(conv_n) + '_' + str(i) + '_' + 'heatmap.png'
    fn3 = str(conv_n) + '_' + str(i) + '_' + 'heatmap_raw.png'
    fn4 = str(i) + '_' + 'CT.png'
    superimposed_img.save(os.path.join(save_dir, fn1))
#    jet_heatmap0.save(os.path.join(save_dir, fn2))
#    jet_heatmap.save(os.path.join(save_dir, fn3))
#    img0.save(os.path.join(save_dir, fn4))


if __name__ == '__main__':
    
    train_img_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/train_img_dir'
    val_save_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/val'
    test_save_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test'
    exval_save_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/exval'
    val_gradcam_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/val/gradcam'
    test_gradcam_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test/gradcam'
    exval_gradcam_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test/gradcam'
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    model_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/model'
    input_channel = 3
    re_size = (192, 192)
    i = 72
    crop = True
    alpha = 0.9
    saved_model = 'ResNet_2021_07_18_06_28_40'
    show_network = False
    conv_n = 'conv5'
    run_type = 'val'

    #---------------------------------------------------------
    # run main function
    #--------------------------------------------------------
    if run_type == 'val':
        save_dir = val_save_dir
    elif run_type == 'test':
        save_dir = test_save_dir

    ## load model and find conv layers   
    model = load_model(os.path.join(model_dir, saved_model))
#    model.summary() 

    list_i = [100, 105, 110, 115, 120, 125]
    for i in list_i:
        image, label, pred_index, y_pred, ID = data(
            input_channel=input_channel,
            i=i,
            val_save_dir=val_save_dir,
            test_save_dir=test_save_dir
            )
        
        conv_list = ['conv2', 'conv3', 'conv4', 'conv5']
        conv_list = ['conv4']
        for conv_n in conv_list:
            if conv_n == 'conv2':
                last_conv_layer = 'conv2_block3_1_conv'
            elif conv_n == 'conv3':
                last_conv_layer = 'conv3_block4_1_conv'
            elif conv_n == 'conv4':
                last_conv_layer = 'conv4_block6_1_conv'
            elif conv_n == 'conv5':
                last_conv_layer = 'conv5_block3_out'

            heatmap = compute_heatmap(
                model=model,
                saved_model=saved_model,
                image=image,
                pred_index=pred_index,
                last_conv_layer=last_conv_layer
                )

            save_gradcam(
                image=image, 
                heatmap=heatmap,
                val_gradcam_dir=val_gradcam_dir,
                test_gradcam_dir=test_gradcam_dir,
                alpha=alpha,
                i=i
                )
        
        print('label:', label)
        print('ID:', ID)
        print('y_pred:', y_pred)
        print('prediction:', pred_index)
        print('conv layer:', conv_n)


            
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
