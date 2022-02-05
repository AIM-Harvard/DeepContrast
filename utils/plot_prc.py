#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-----------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_recall_curve


# ----------------------------------------------------------------------------------
# precision recall curve
# ----------------------------------------------------------------------------------
def plot_prc(save_dir, y_true, y_pred, level, color):

    precision = dict()
    recall    = dict()
    threshold = dict()
    prc_auc   = []

    precision, recall, threshold = precision_recall_curve(y_true, y_pred) 
    RP_2D = np.array([recall, precision])
    RP_2D = RP_2D[np.argsort(RP_2D[:, 0])]
    #prc_auc.append(auc(RP_2D[1], RP_2D[0]))
    prc_auc = auc(RP_2D[1], RP_2D[0])
    prc_auc = np.around(prc_auc, 3)
	#print('PRC AUC:', prc_auc)   
    #prc_auc = auc(precision, recall)
    print(prc_auc)
    #prc_auc = 1

    fn = 'prc' + '_' + str(level) + '.png'   
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.plot(
             recall,
             precision,
             color=color,
             linewidth=3,
             label='AUC %0.3f' % prc_auc
             )
    plt.xlim([0, 1.03])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=1.03, color='k', linewidth=4) 
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.xlabel('recall', fontweight='bold', fontsize=16)
    plt.ylabel('precision', fontweight='bold', fontsize=16)
    plt.legend(loc='lower left', prop={'size': 16, 'weight': 'bold'}) 
    plt.grid(True)
#    plt.tight_layout(pad=0.2, h_pad=None, w_pad=None, rect=None)
#    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(save_dir, fn), format='png', dpi=600)
    #plt.show()
    plt.close()
    
    return prc_auc



    

     
