# Source Code

Here follows a brief description of how the source code is organised and what are the different steps of the processing.

Additional information regarding the output of the pipeline (e.g., where the results of a certain step will be exported) are found in the markdown file under the `data` directory of the repository.

## Model Prediction

The get data step takes care of the following operations:

1. Sample data (head and neck CT and chest CT) can be found from foler "datasets";
2. Trained models based on EfficientNetB4 for head and neck CT ("EffNet_HeadNeck.h5") and chest CT ("EffNet_Chest.h5") can be found from foler "models"; 
3. Data preprocessing for either head and neck or chest CT scans including respacing, registration and cropping (`data_prepro.py`);
4. Predict IV contrast for head and neck or chest CT scans and save results to csv file (`model_pred.py`);

The get data step can be run by executing:

```
python run_prediction.py
```

## Model Development and Test

### Step 1: Get Data

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

1. Create data generators, including augmentation for training and validation dataset (`data_gen_flow.py`);
2. Generate desired CNN models, including simple CNN model, EfficientNetB4 model, ResNet101V2 model, Inception3 model, and Transfer Learning model (`get_model.py`);
3. Train model and save training results and hyperparaters to txt file (`train_model.py`);

The train step can be run by executing:

```
python run_step2_train.py
```

### Step 3: Test

The test step takes care of the following operations:

1. Evaluate model performance on internal validation dataset (head and neck CT) and external test dataset (head and neck CT) (`evaluate_model.py`);
2. Generate statistcal results (accuracy, ROC-AUC, sensitivity, specificity, F1-score) and plots (confusion matrix, ROC curve, precision-recall curve) ('get_stats_plots.py);

The test step can be run by executing:

```
python run_step3_test.py
```

### Step 4: External Validation

1. Preprocess chest CT data and prepare data for the CNN model input (`exval_dataset.py`);
2. Fine tune previsouly trained model with chest CT data (`finetune_model.py`);
3. Evaluate fine-tuned model with internal validation dataset (chest CT) and external test dataset (chest CT) (`evaluate_model.py`);
4. Generate statistcal results (accuracy, ROC-AUC, sensitivity, specificity, F1-score) and plots (confusion matrix, ROC curve, precision-recall curve) ('get_stats_plots.py);

The external validation step can be run by executing:

```
python run_step4_exval.py
```

