# AMLS_20-21_SN20108057

### Description

A brief description of the organization of the project is presented below in a logical order: 

1. Step 1 - Feature extraction: face_landmarks.py, canny_edge_detection.py
1. Step 2 - Data pre-processing: data_preprocessing.py
1. Step 3 - Model selection: model_selection.py
1. Step 4 - Classification (train, validate, and test using the final models): task_A1.py, task_A2.py, task_B1.py, task_B2.py  
1. Step 5 - Result display: result_display.py, plot_learning_curve.py
1. Step 6 - Project execution: main.py

### Prerequisites

In your Python 3.6 environment or machine, from the route directory of where you
cloned this project, install the required packages:

```
dlib==19.21.1
hypopt==1.0.9
matplotlib==3.3.3
numpy==1.19.4
opencv-python==4.4.0.46
pandas==1.1.5
scikit-learn==0.23.2
tensorflow==1.10.0
Keras==2.2.4
```

The shape_predictor_68_face_landmarks.dat file, which is used in the face_landmarks.py script, can be downloaded through the following link: 
https://drive.google.com/file/d/1gtCmJpFhABFXkRL6XPNV3N427I2w9Jxx/view?usp=sharing

Add the specified datasets into the same folder of where you cloned this project. 

### Usage

The role of each file in this project is illustrated as follows:

* The main.py script contains the main body of this project, which is run only to train, validate, and test the optimal machine learning model selected for the specified four tasks. 
* The **task_A1.py** script implements binary classification for Task A1.
* The **task_A2.py** script implements binary classification for Task A2.
* The **task_B1.py** script implements multiclass classification for Task B1.
* The **task_B2.py** script implements multiclass classification for Task B2.
* The **result_display.py** script acquires the corresponding classification performance metrics on training, validation, and test datasets for a model and prints to console.
* The **plot_learning_curve.py** script plots learning curves using cross-validation.
* The **model_selection.py** script performs model selection based on grid search hyper-parameter optimization.
* The **data_preprocessing.py** script carries out data pre-processing of the raw image data from the two datasets.
* The **canny_edge_detection.py** script performs Canny edge detection.
* The **face_landmarks.py** script conducts the corresponding feature extraction approaches for all the tasks.

