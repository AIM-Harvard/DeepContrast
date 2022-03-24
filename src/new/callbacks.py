import os
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def callbacks(log_dir):
  
    check_point = ModelCheckpoint(
        filepath=os.path.join(log_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5'),
        monitor='val_acc',
        verbose=1,
        save_best_model_only=True,
        save_weights_only=True,
        mode='max'
        )
  
    tensor_board = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None
        )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False
        )
                               
    my_callbacks = [
        #ModelSave(),
        early_stopping,
        #LearningRateScheduler(shcheduler),
        #check_point,
        tensor_board
        ]

    return my_callbacks


    

    
