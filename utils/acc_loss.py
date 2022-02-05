import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

def acc_loss(run_type, saved_model, output_dir, save_dir):

    ### determine if this is train or test
    if run_type == 'val':
        fn = 'df_val_pred.p'
    if run_type == 'test':
        fn = 'df_test_pred.p'
     
    df_sum = pd.read_pickle(os.path.join(save_dir, fn))
    y_true = df_sum['label'].to_numpy()
    y_pred = df_sum['y_pred_class'].to_numpy()
    
    ### acc and loss
    model = load_model(os.path.join(output_dir, saved_model))
    score = model.evaluate(x_val, y_val)
    loss = np.around(score[0], 3)
    acc  = np.around(score[1], 3)
    print('loss:', loss)
    print('acc:', acc)

    return acc, loss
