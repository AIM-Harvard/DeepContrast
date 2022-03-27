import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--root_dir', default='/mnt/aertslab/USERS/Zezhong/contrast_detection/GitHub_Test', 
                        type=str, help='Root path')
    # head and neck dataset 
    parser.add_argument('--HN_data', default='HeadNeck/raw_image', type=str, help='Raw image path')
    parser.add_argument('--HN_pre_data', default='HeadNeck/pre_image', type=str, help='Preprocessed image path')
    parser.add_argument('--HN_label', default='HeadNeck/label', type=str, help='Label path')
    parser.add_argument('--HN_out', default='HeadNeck/out', type=str, help='Results output path')
    parser.add_argument('--HN_pro_data', default='HeadNeck/pro_data', type=str, help='Processed data path')
    parser.add_argument('--HN_model', default='HeadNeck/out/model', type=str, help='Results output path')
    parser.add_argument('--HN_log', default='HeadNeck/out/log', type=str, help='Log data path')
    parser.add_argument('--HN_train', default='HeadNeck/out/train', type=str, help='Train results path')
    parser.add_argument('--HN_val', default='HeadNeck/out/val', type=str, help='Validation results path')
    parser.add_argument('--HN_test', default='HeadNeck/out/test', type=str, help='Test results path')
    # chest dataset
    parser.add_argument('--CH_data', default='Chest/raw_image', type=str, help='Raw image path')
    parser.add_argument('--CH_pre_data', default='Chest/pre_image', type=str, help='Preprocessed image path')
    parser.add_argument('--CH_label', default='Chest/label', type=str, help='Label path')
    parser.add_argument('--CH_out', default='Chest/out', type=str, help='Results output path')
    parser.add_argument('--CH_pro_data', default='Chest/pro_data', type=str, help='Processed data path')
    parser.add_argument('--CH_model', default='Chest/out/model', type=str, help='Results output path')
    parser.add_argument('--CH_log', default='Chest/out/log', type=str, help='Log data path')
    parser.add_argument('--CH_train', default='Chest/out/train', type=str, help='Train results path')
    parser.add_argument('--CH_val', default='Chest/out/val', type=str, help='Validation results path')
    parser.add_argument('--CH_test', default='Chest/out/test', type=str, help='Test results path')

    # data preprocessing
    parser.add_argument('--preprocess_data', action='store_true', help='If true, test is performed.')
    parser.set_defaults(preprocess_data=True)
    parser.add_argument('--manual_seed', default=1234, type=int, help='seed')
    parser.add_argument('--HN_label_file', default='HN_label.csv', type=str, help='Head and neck data label file')
    parser.add_argument('--CH_label_file', default='CH_label.csv', type=str, help='Chest data label file')
    parser.add_argument('--new_spacing', default=(1, 1, 3), type=float, help='new spacing size')
    parser.add_argument('--HN_crop_shape', default=[192, 192, 100], type=float, help='HeadNeck crop image shape')
    parser.add_argument('--CH_crop_shape', default=[192, 192, 140], type=float, help='Chest crop image shape')
    parser.add_argument('--run_type', default=None, type=str, help='Used run type (train | val | test | tune)')
    parser.add_argument('--input_channel', default=3, type=int, help='Input channel (3 | 1)')
    parser.add_argument('--norm_type', default='np_clip', type=str, 
                        help='CT image normalization (np_clip | np_linear')
    parser.add_argument('--HN_slice_range', default=range(17, 83), type=int, help='Head and neck axial slice range')
    parser.add_argument('--CH_slice_range', default=range(50, 120), type=int, help='Chest axial slice range')
    parser.add_argument('--interp', default='linear', type=str, help='Interpolation for respacing')
    parser.add_argument('--data_exclude', default=None, type=str, help='excluding data')
    
    # train model
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epoch', default=1, type=int, help='Epoch')
    parser.add_argument('--activation', default='sigmoid', type=str, help='Activation function on last layer')
    parser.add_argument('--loss_function',  default='binary_crossentropy', type=str,
                        help='loss function (binary_crossentropy | crossentropy)')
    parser.add_argument('--optimizer_function', default='adam', type=str, help='optmizer function')
    parser.add_argument('--run_model', default='EffNetB4', type=str,
                        help='run model (EffNetB4 | ResNet101V2 | InceptionV3 | CNN)')
    parser.add_argument('--input_shape', default=(192, 192, 3), type=int, help='Input shape')
    parser.add_argument('--freeze_layer', default=None, type=str, help='Freeze layer to train')
    parser.add_argument('--transfer', action='store_true', help='If true, transfer learning is performed.')
    parser.set_defaults(transfer=False)

    # evalute model   
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--stats_plots', action='store_true', help='If true, plots and statistics is performed.')
    parser.set_defaults(stats_plots=True)
    parser.add_argument('--thr_img', default=0.5, type=float, help='threshold to decide class on image level')
    parser.add_argument('--thr_prob', default=0.5, type=float, help='threshold to decide class on patient level')
    parser.add_argument('--thr_pos', default=0.5, type=float, help='threshold to decide class on patient level')
    parser.add_argument('--n_bootstrap', default=1000, type=int, help='n times of bootstrap to calcualte 95% CI')
    parser.add_argument('--saved_model', default='EffNetB4', type=str, help='saved model name')    

    # fine tune model
    parser.add_argument('--get_CH_data', action='store_true', help='If true, get_data is performed.')
    parser.set_defaults(get_Ch_data=True)
    parser.add_argument('--fine_tune', action='store_true', help='If true, fine_tune is performed.')
    parser.set_defaults(fine_tune=True)
    parser.add_argument('--tuned_model', default='Tuned_EffNetB4', type=str, help='tuned model')    
    
    
    args = parser.parse_args()

    return args
