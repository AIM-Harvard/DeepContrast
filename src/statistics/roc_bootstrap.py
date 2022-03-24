import os
import numpy as np
import pandas as pd
import glob
from sklearn.utils import resample
import scipy.stats as ss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def mean_CI(data):

    mean = np.mean(np.array(data))
    CI = ss.t.interval(
        alpha=0.95,
        df=len(data)-1,
        loc=np.mean(data),
        scale=ss.sem(data)
        )
    lower = CI[0]
    upper = CI[1]

    return mean, lower, upper


def roc_bootstrap(bootstrap, y_true, y_pred):

    AUC  = []
    THRE = []
    TNR  = []
    TPR  = []
    for j in range(bootstrap):
        #print("bootstrap iteration: " + str(j+1) + " out of " + str(n_bootstrap))
        index = range(len(y_pred))
        indices = resample(index, replace=True, n_samples=int(len(y_pred)))
        fpr, tpr, thre = roc_curve(y_true[indices], y_pred[indices])
        q = np.arange(len(tpr))
        roc = pd.DataFrame(
            {'fpr' : pd.Series(fpr, index=q),
             'tpr' : pd.Series(tpr, index=q),
             'tnr' : pd.Series(1 - fpr, index=q),
             'tf'  : pd.Series(tpr - (1 - fpr), index=q),
             'thre': pd.Series(thre, index=q)}
             )
        ### calculate optimal TPR, TNR under uden index
        roc_opt = roc.loc[(roc['tpr'] - roc['fpr']).idxmax(),:]
        AUC.append(roc_auc_score(y_true[indices], y_pred[indices]))
        TPR.append(roc_opt['tpr'])
        TNR.append(roc_opt['tnr'])
        THRE.append(roc_opt['thre'])
    ### calculate mean and 95% CI
    AUCs  = np.around(mean_CI(AUC), 3)
    TPRs  = np.around(mean_CI(TPR), 3)
    TNRs  = np.around(mean_CI(TNR), 3)
    THREs = np.around(mean_CI(THRE), 3)
    #print(AUCs)
    ### save results into dataframe
    stat_roc = pd.DataFrame(
        [AUCs, TPRs, TNRs, THREs],
        columns=['mean', '95% CI -', '95% CI +'],
        index=['AUC', 'TPR', 'TNR', 'THRE']
        )

    return stat_roc
