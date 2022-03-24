import os
import numpy as np
import pandas as pd
from datetime import datetime
from time import localtime, strftime





def write_txt(run_type, proj_dir, loss, acc, cms, cm_norms, reports, prc_aucs, 
              roc_stats, run_model, saved_model, epoch, batch_size, lr): 

    """
    write model training, val, test results to txt files;

    Args:
        loss {float} -- loss value;
        acc {float} -- accuracy value;
        cms {list} -- list contains confusion matrices;
        cm_norms {list} -- list containing normalized confusion matrices;
        reports {list} -- list containing classification reports;
        prc_aucs {list} -- list containing prc auc;
        roc_stats {list} -- list containing ROC statistics;
        batch_size {int} -- batch size for data loading;
        epoch {int} -- training epoch;
        out_dir {path} -- path for output files;
     
    Returns:
        save model results and parameters to txt file
    
    """

    train_dir = os.path.join(proj_dir, 'HeadNeck/output/train')
    val_dir = os.path.join(out_dir, 'HeadNeck/output/val')
    test_dir = os.path.join(out_dir, 'HeadNeck/output/test')
    tune_dir = os.path.join(out_dir, 'Chest/output/tune')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(tune_dir):
        os.makedirs(tune_dir)

    if run_type == 'train':
        log_fn = 'train_logs.text'
        save_dir = train_dir
        write_path = os.path.join(save_dir, log_fn)
        with open(write_path, 'a') as f:
            f.write('\n-------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\nval acc: %s' % acc)
            f.write('\nval loss: %s' % loss)
            f.write('\nrun model: %s' % run_model)
            f.write('\nsaved model: %s' % saved_model)
            f.write('\nepoch: %s' % epoch)
            f.write('\nlearning rate: %s' % lr)
            f.write('\nbatch size: %s' % batch_size)
            f.write('\n')
            f.close()
        print('successfully save train logs.')
    else:
        if run_type == 'val':
            log_fn = 'val_logs.text'
            save_dir = val_dir
        elif run_type == 'test':
            log_fn = 'test_logs.text'
            save_dir = test_dir
        elif run_type == 'tune':
            log_fn = 'tune_logs.text'
            save_dir = tune_dir
        
        write_path = os.path.join(save_dir, log_fn)
        with open(write_path, 'a') as f:
            f.write('\n------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\nval accuracy: %s' % acc)
            f.write('\nval loss: %s' % loss)
            f.write('\nprc image: %s' % prc_aucs[0])
            f.write('\nprc patient prob: %s' % prc_aucs[1])
            f.write('\nprc patient pos: %s' % prc_aucs[2])
            f.write('\nroc image:\n %s' % roc_stats[0])
            f.write('\nroc patient prob:\n %s' % roc_stats[1])
            f.write('\nroc patient pos:\n %s' % roc_stats[2])
            f.write('\ncm image:\n %s' % cms[0])
            f.write('\ncm image:\n %s' % cm_norms[0])
            f.write('\ncm patient prob:\n %s' % cms[1])
            f.write('\ncm patient prob:\n %s' % cm_norms[1])
            f.write('\ncm patient pos:\n %s' % cms[2])
            f.write('\ncm patient pos:\n %s' % cm_norms[2])
            f.write('\nreport image:\n %s' % reports[0])
            f.write('\nreport patient prob:\n %s' % reports[1])
            f.write('\nreport patient pos:\n %s' % reports[2])
            f.write('\nrun model: %s' % run_model)
            f.write('\nsaved model: %s' % saved_model)
            f.write('\nepoch: %s' % epoch)
            f.write('\nlearning rate: %s' % lr)
            f.write('\nbatch size: %s' % batch_size)
            f.write('\n')
            f.close()    
        print('successfully save logs.')
    
