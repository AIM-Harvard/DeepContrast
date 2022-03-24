import os
import numpy as np
import pydot
import pydotplus
import graphviz
from tensorflow.keras.utils import plot_model
from models.simple_cnn import simple_cnn
from models.EfficientNet import EfficientNet
from models.ResNet import ResNet
from models.Inception import Inception
from models.VGGNet import VGGNet
from models.TLNet import TLNet




def get_model(out_dir, run_model, activation, input_shape=(192, 192, 3), 
              freeze_layer=None, transfer=False):
    
    """
    generate cnn models

    Args:
        run_model {str} -- choose specific CNN model type;
        activation {str or function} -- activation function in last layer: 'sigmoid', 'softmax', etc;
    
    Keyword args:i
        input_shape {np.array} -- input data shape;
        transfer {boolean} -- decide if transfer learning;
        freeze_layer {int} -- number of layers to freeze;
    
    Returns:
        deep learning model;
    
    """
    

    train_dir = os.path.join(out_dir, 'train')
    if not os.path.exists(train_dir):
        os.path.mkdir(train_dir)

    if run_model == 'cnn':
        my_model = simple_cnn(
            input_shape=input_shape,
            activation=activation,
            )
    elif run_model == 'ResNet101V2':
        my_model = ResNet(
            resnet='ResNet101V2',  #'ResNet50V2',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation,
            )
    elif run_model == 'EffNetB4':
        my_model = EfficientNet(
            effnet='EffNetB4',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )
    elif run_model == 'TLNet':
        my_model = TLNet(
            resnet='ResNet101V2',
            input_shape=input_shape,
            activation=activation
            )
    elif run_model == 'InceptionV3':
        my_model = Inception(
            inception='InceptionV3',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )

    print(my_model)
    
    # plot cnn architectures and save png    
    fn = os.path.join(train_dir, str(run_model) + '.png')
    plot_model(
        model=my_model,
        to_file=fn,
        show_shapes=True,
        show_layer_names=True
        )


    return my_model




