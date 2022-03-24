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
from go_model.callbacks import callbacks
from utils.plot_train_curve import plot_train_curve
from tensorflow.keras.optimizers import Adam
from utils.write_txt import write_txt



def train_model(log_dir, model_dir, model, run_model, train_gen, val_gen, x_val, y_val, batch_size, epoch, 
                optimizer, loss_function, lr): 

    """
    train model

    Args:
        model {cnn model} -- cnn model;
        run_model {str} -- cnn model name;
        train_gen {Keras data generator} -- training data generator with data augmentation;
        val_gen {Keras data generator} -- val data generator;
        x_val {np.array} -- np array for validation data;
        y_val {np.array} -- np array for validation label;
        batch_size {int} -- batch size for data loading;
        epoch {int} -- training epoch;
        out_dir {path} -- path for output files;
        opt {str or function} -- optimized function: 'adam';
        loss_func {str or function} -- loss function: 'binary_crossentropy';
        lr {float} -- learning rate;

    Returns:
        training accuracy, loss, model
    
    """

    ## compile model
    print('complie model')
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['acc']
        )
	
    ## call back functions
    my_callbacks = callbacks(log_dir)
    
    ## fit models
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n//batch_size,
        epochs=epoch,
        validation_data=val_gen,
        #validation_data=(x_val, y_val),
        validation_steps=val_gen.n//batch_size,
        #validation_steps=y_val.shape[0]//batch_size,
        verbose=2,
        callbacks=my_callbacks,
        validation_split=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0
        )
    
    ## valudation acc and loss
    score = model.evaluate(x_val, y_val)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('val loss:', loss)
    print('val acc:', acc)

    ## save final model
    saved_model = str(run_model) + '_' + str(strftime('%Y_%m_%d_%H_%M_%S', localtime()))
    model.save(os.path.join(model_dir, saved_model))
    print(saved_model)
    
    ## save validation results to txt file 
    write_txt(
        run_type='train',
        out_dir=out_dir,
        loss=1,
        acc=1,
        cms=None,
        cm_norms=None,
        reports=None,
        prc_aucs=None,
        roc_stats=None,
        run_model=run_model,
        saved_model=saved_model,
        epoch=epoch,
        batch_size=batch_size,
        lr=lr
        )

    


    

    
