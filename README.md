# DeepContrast

Keras Implementation of the article "[Deep learning-based detection of intravenous contrast in computed tomography scans](https://arxiv.org/pdf/2110.08424.pdf)", codes and pretrained models.

## Implemented models

 - Simple CNN
 - InceptionV3
 - ResNet101v2
 - Transfer Learning based on ResNet101v2
 - EfficientNetB4

## Results

<p align="center"><img src="https://github.com/AIM-Harvard/DeepContrast/blob/main/src/utils/Table.png" align="middle" width="750" title="Results of Head and Neck CT scans" /></p>

## Repository Structure

The DeepContrast repository is structured as follows:

All the source code to run the deep-learning-based pipeline is found under the src folder.
Four sample subjects' CT data and the associated data label, as well as all the models weights necessary to run the pipeline, are stored under the data folder.
Additional details on the content of the subdirectories and their structure can be found in the markdown files stored in the former.

## Requirements

* Python 3.8
* TensorFlow 2.4

## Set-up

This code was developed and tested using Python 3.8.5.

For the code to run as intended, all the packages under requirements.txt should be installed. In order not to break previous installations and ensure full compatibility, it's highly recommended to create a virtual environment to run the DeepContrast pipeline in. Here follows an example of set-up using python virtualenv:

* install python's virtualenv
```
sudo pip install virtualenv
```
* parse the path to the python3 interpreter
```
export PY2PATH=$(which python3)
```
* create a virtualenv with such python3 interpreter named "venv"
(common name, already found in .gitignore)
```
virtualenv -p $PY2PATH venv 
```
* activate the virtualenv
```
source venv/bin/activate
```
At this point, (venv) should be displayed at the start of each bash line. Furthermore, the command which python3 should return a path similar to /path/to/folder/venv/bin/python3. Once the virtual environment is activated:

* once the virtualenv is activated, install the dependencies
```
pip install -r requirements.txt
```
At this stage, everything should be ready for the data to be processed by the DeepContrast pipeline. Additional details can be found in the markdown file under src.

The virtual environment can be deactivated by running:
```
deactivate
```

## Model Inference

The get data step takes care of the following operations:

1. Run the code (`run_inference.py`) under the root path of "src" for model inference on sample data;
2. Specify "HeadNeck" or "Chest" to predict sample data from head and neck or chest CT scan;
3. Data preprocessing for either head and neck or chest CT scans including respacing, registration and cropping (`data_prepro.py`);
4. Predict IV contrast for head and neck or chest CT scans and save results to csv file (`model_pred.py`);

The model inference can be run by executing:

```
python run_inference.py --HeadNeck
```
or

```
python run_inference.py --Chest
```
## Model Development and Test

### Step 1: Get Data

Download head-and-neck and chest sample CT scans and IV contrast labels from [here](https://drive.google.com/drive/folders/1xXU3GoM4_5CnzPB_8eD3Zjmk6Ye1y7ad?usp=sharing).

Assume the structure of data directories is the following:
```misc
~/
  HeadNeck/
    raw_image/
      nrrd files
    label/
      csv file
  Chest/
    raw_image/
      nrrd files
    label/
      csv file
```
The get data step takes care of the following operations:
1. Data preprocessing for head and neck CT scan including respacing, registration and cropping (`preprocess_data.py`);
2. Create dataframe to contain data paths, patient IDs and labels on the
    patient level (`get_pat_dataset.py`);
3. Get stacked axial image slices, labels and IDs on the image level for train, validation, test datasets (`get_img_dataset.py`);

The get data step can be run by executing:

```
python run_step1_data.py
```
### Step 2: Train

The train step takes care of the following operations:

1. Create data generators, including augmentation for training and validation dataset (`data_generator.py`);
2. Generate desired CNN models, including simple CNN model, EfficientNetB4 model, ResNet101V2 model, Inception3 model, and Transfer Learning model (`get_model.py`);
3. Train model and save training results and hyperparaters to txt file (`train_model.py`);

The train step can be run by executing:

```
python run_step2_train.py
```

### Step 3: Test

The test step takes care of the following operations:

1. Evaluate model performance on internal validation dataset (head and neck CT) and external test dataset (head and neck CT) (`evaluate_model.py`);
2. Generate statistcal results (accuracy, ROC-AUC, sensitivity, specificity, F1-score) and plots (confusion matrix, ROC curve, precision-recall curve) (`get_stats_plots.py`);

The test step can be run by executing:

```
python run_step3_test.py
```

### Step 4: Finetune Model for Chest CT Data

1. Preprocess chest CT data and prepare data for the CNN model input (`tune_dataset.py`);
2. Fine tune previsouly trained model with chest CT data (`tune_model.py`);
3. Evaluate fine-tuned model with internal validation dataset (chest CT) and external test dataset (chest CT) (`evaluate_model.py`);
4. Generate statistcal results (accuracy, ROC-AUC, sensitivity, specificity, F1-score) and plots (confusion matrix, ROC curve, precision-recall curve) (`get_stats_plots.py`);

The external validation step can be run by executing:

```
python run_step4_tune.py
```

## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2110.08424,
  doi = {10.48550/ARXIV.2110.08424},
  url = {https://arxiv.org/abs/2110.08424}, 
  author = {Ye Z, Qian JM, Hosny A, Zeleznik R, Plana D, Likitlersuang J, Zhang Z, Mak RH, Aerts HJWL, and Kann BH},
  title = {Deep learning-based detection of intravenous contrast in computed tomography scans},
  publisher = {arXiv},
  year = {2021}, 
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```

## Acknowledgements

Code development, testing, refactoring and documentation: ZY, BHK.

## Disclaimer

The code and data of this repository are provided to promote reproducible research. They are not intended for clinical care or commercial use.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
