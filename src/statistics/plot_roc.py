import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def plot_roc(save_dir, y_true, y_pred, level, color):
    
    fpr       = dict()
    tpr       = dict()
    roc_auc   = dict()
    threshold = dict()

    ### calculate auc
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_auc = np.around(roc_auc, 3)
    #print('ROC AUC:', roc_auc)
    
    fn = 'roc'+ '_' + str(level) + '.png'
    ### plot roc
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.plot(fpr, tpr, color=color, linewidth=3, label='AUC %0.3f' % roc_auc)
    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=-0.03, color='k', linewidth=4)
    ax.axvline(x=1, color='k', linewidth=4)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.xlabel('1 - Specificity', fontweight='bold', fontsize=16)
    plt.ylabel('Sensitivity', fontweight='bold', fontsize=16)
    plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'})
    plt.grid(True)
#    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.savefig(os.path.join(save_dir, fn), format='png', dpi=600)
    #plt.show()
    plt.close()

    return roc_auc
