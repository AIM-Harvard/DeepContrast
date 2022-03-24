import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--root_dir', 
                        default='/mnt/aertslab/USERS/Zezhong/contrast_detection/GitHub_Test', 
                        type=str, 
                        help='Root path')
    parser.add_argument('--data_dir', 
                        default='raw_image/HN', 
                        type=str,
                        help='Raw image path')
    parser.add_argument('--pre_data_dir', 
                        default='pre_image/HN', 
                        type=str, 
                        help='Preprocessed image path')
    parser.add_argument('--label_dir', 
                        default='label', 
                        type=str, 
                        help='Label path')
    parser.add_argument('--out_dir', 
                        default='output', 
                        type=str, 
                        help='Results output path')
    parser.add_argument('--pro_data', 
                        default='pro_data', 
                        type=str, 
                        help='Processed data path')
    parser.add_argument('--model_dir', 
                        default='output/model', 
                        type=str, 
                        help='Results output path')
    parser.add_argument('--log_dir', 
                        default='output/log', 
                        type=str, 
                        help='Log data path')
    parser.add_argument('--train_dir', 
                        default='output/train', 
                        type=str, 
                        help='Train results path')
    parser.add_argument('--val_dir', 
                        default='output/val', 
                        type=str, 
                        help='Validation results path')
    parser.add_argument('--test_dir',
                        default='output/test',
                        type=str,
                        help='Test results path')
    
    # data preprocessing
    parser.add_argument('--new_spacing',
                        default=(1, 1, 3),
                        type=float,
                        help='new spacing size')
    parser.add_argument('--data_exclude',
                        default=None,
                        type=str,
                        help='Exclude data')
    parser.add_argument('--crop_shape',
                        default=[192, 192, 100],
                        type=float,
                        help='Crop image shape')
    parser.add_argument('--run_type',
                        default=None,
                        type=str,
                        help='Used run type (train | val | test | pred)')
    parser.add_argument('--input_channel',
                        default=3,
                        type=int,
                        help='Input channel (pre_trained: 3, train from scratch: 1)')
    parser.add_argument('--norm_type',
                        default='np_clip',
                        type=str,
                        help='CT image normalization (np_clip | np_linear')
    parser.add_argument('--slice_range',
                        default=range(17, 83),
                        type=int,
                        help='Axial slice range to keep the image')
    parser.add_argument('--interp',
                        default='linear',
                        type=str,
                        help='Interpolation for respacing')

    # train model
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='Batch size')
    parser.add_argument('--lr',
                        default=1e-5,
                        type=float,
                        help='learning rate')
    parser.add_argument('--epoch',
                        default=1,
                        type=int,
                        help='Epoch')
    parser.add_argument('--activation',
                        default='sigmoid',
                        type=str,
                        help='Activation function on last layer')
    parser.add_argument('--loss_function',
                        default='binary_crossentropy',
                        type=str,
                        help='loss function (binary_crossentropy | crossentropy)')
    parser.add_argument('--optimizer_function',
                        default='adam',
                        type=str,
                        help='optmizer function')
    parser.add_argument('--run_model',
                        default='EffNetB4',
                        type=str,
                        help='run model (EffNetB4 | ResNet101V2 | InceptionV3 | CNN)')
    parser.add_argument('--input_shape',
                        default=(192, 192, 3),
                        type=int,
                        help='Input shape')
    parser.add_argument('--freeze_layer',
                        default=None,
                        type=str,
                        help='Freeze layer to train')
    parser.add_argument('--transfer',
                        default=False,
                        type=boolean,
                        help='Transfer learnong or not (True | False')

    # evalute model                        
    parser.add_argument('--thr_img',
                        default=0.5,
                        type=float,
                        help='threshold to decide positive class on image level')
    parser.add_argument('--thr_prob',
                        default=0.5,
                        type=float,
                        help='threshold to decide positive class on patient level')
    parser.add_argument('--thr_pos',
                        default=0.5,
                        type=float,
                        help='threshold to decide positive class on patient level')
    parser.add_argument('--n_bootstrap',
                        default=1000,
                        type=int,
                        help='n times of bootstrapping to calcualte 95% CI of AUC')
    parser.add_argument('--saved_model',
                        default='EffNetB4',
                        type=str,
                        help='saved model name')    

    # fine tune model
    
    # others 
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)

    args = parser.parse_args()

    return args
