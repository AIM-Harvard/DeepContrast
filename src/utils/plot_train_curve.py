import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def plot_train_curve(output_dir, epoch, fn, history):
    
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    n_epoch = list(range(epoch))
    ## accuracy curves
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(n_epoch, train_acc, label='Train Acc')
    plt.plot(n_epoch, val_acc, label='Val Acc')
    plt.legend(loc='lower right')
    plt.title('Train and Tune Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ## loss curves
    plt.subplot(2, 2, 2)
    plt.plot(n_epoch, train_loss, label='Train Loss')
    plt.plot(n_epoch, val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Train and Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(os.path.join(output_dir, fn))
    plt.close()
