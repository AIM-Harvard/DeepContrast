import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob
from collections import Counter
from datetime import datetime
from time import localtime, strftime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model



def finetune_model(out_dir, proj_dir, HN_model, batch_size, epoch, 
                   freeze_layer, input_channel=3): 

    """
    Fine tune head anc neck model using chest CT data;

    Args:
        out_dir {path} -- path to main output folder;
        proj_dir {path} -- path to main project folder;
        saved_model {str} -- saved model name;
        batch_size {int} -- batch size to load the data;
        epoch {int} -- running epoch to fine tune model, 10 or 20;
        freeeze_layer {int} -- number of layers in HN model to freeze durding fine tuning;
    i
    Keyword args:
        input_channel {int} -- image channel: 1 or 3;
    
    Returns:
        Finetuned model for chest CT.
    
    """

    model_dir = os.path.join(out_dir, 'model')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(pro_data_dir):
        os.mkdir(pro_data_dir)

    ### load train data
    if input_channel == 1:
        fn = 'exval1_arr1.npy'
    elif input_channel == 3:
        fn = 'exval1_arr1.npy'
    x_train = np.load(os.path.join(pro_data_dir, fn))
    ### load train labels
    train_df = pd.read_csv(os.path.join(pro_data_dir, 'exval1_img_df1.csv'))
    y_train  = np.asarray(train_df['label']).astype('int').reshape((-1, 1))
    print("sucessfully load data!")

    ## load saved model
    model = load_model(os.path.join(model_dir, HN_model))
    model.summary()

    ### freeze specific number of layers
    if freeze_layer != None:
        for layer in model.layers[0:freeze_layer]:
            layer.trainable = False
        for layer in model.layers:
            print(layer, layer.trainable)
    else:
        for layer in model.layers:
            layer.trainable = True
    model.summary()

    ### fit data into dnn models
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=None,
        verbose=1,
        callbacks=None,
        validation_split=0.3,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0
        )
    
#    ### valudation acc and loss
#    score = model.evaluate(x_val, y_val)
#    loss = np.around(score[0], 3)
#    acc  = np.around(score[1], 3)
#    print('val loss:', loss)
#    print('val acc:', acc)

    #### save final model
    run_model = saved_model.split('_')[0].strip()
    model_fn = 'Tuned' + '_' + str(run_model) + '_' + \
               str(strftime('%Y_%m_%d_%H_%M_%S', localtime()))
    model.save(os.path.join(model_dir, model_fn))
    tuned_model = model
    print('fine tuning model complete!!')
    print('saved fine-tuned model as:', model_fn)

    return tuned_model, model_fn

    

    
