import data_preprocessing as dp
from model_selection import img_baggingClassifier
from model_selection import img_boostingClassifier
from model_selection import img_SVM
from model_selection import img_KNN
from model_selection import img_logRegression
from model_selection import img_MLP
from model_selection import img_randomForest
from model_selection import img_CNN

from sklearn import metrics
from task_A1 import model_A1
from task_A2 import model_A2
from task_B1 import model_B1
from task_B2 import model_B2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
import plot_learning_curve as lc
from hypopt import GridSearch
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# ======================================================================================================================
# Data preprocessing
# ======
# Task A
# ======
X_train, y_train, X_val, y_val, X_test, y_test, \
X_train_emo, y_train_emo, X_val_emo, y_val_emo, X_test_emo, y_test_emo = dp.data_preprocessing_task_a()

# ======
# Task B
# ======
X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f, \
X_train_eye, y_train_eye, X_val_eye, y_val_eye, X_test_eye, y_test_eye = dp.data_preprocessing_task_b()
# ======================================================================================================================
# Train, validate and test the model for each task
# =======
# Task A1
# =======
# Train model based on the training set
# Fine-tune the model based on the validation set
# Generalise the model based on the test set
# Compute and print the final results to the console
acc_A1_test = model_A1(X_train, y_train[:,0], X_val, y_val[:,0], X_test, y_test[:,0])

# =======
# Task A2
# =======
# Train model based on the training set
# Fine-tune the model based on the validation set
# Generalise the model based on the test set
# Compute and print the final results to the console
acc_A2_test = model_A2(X_train_emo, y_train_emo[:,0], X_val_emo, y_val_emo[:,0], X_test_emo, y_test_emo[:,0])

# =======
# Task B1
# =======
# Train model based on the training set
# Fine-tune the model based on the validation set
# Generalise the model based on the test set
# Compute and print the final results to the console
acc_B1_test = model_B1(X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f)

# =======
# Task B2
# =======
# Train model based on the training set
# Fine-tune the model based on the validation set
# Generalise the model based on the test set
# Compute and print the final results to the console
acc_B2_test = model_B2(X_train_eye, y_train_eye, X_val_eye, y_val_eye, X_test_eye, y_test_eye)

# ======================================================================================================================
# Grid Search
# =======
# Task A1
# =======
# Pre-process data: Normalisation for logistic regression and MLP
#min_max_scaler = MinMaxScaler()  # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
#X_train_normal = min_max_scaler.fit_transform(X_train)
#X_val_normal = min_max_scaler.transform(X_val)
#X_test_normal = min_max_scaler.transform(X_test)

#pred = img_SVM(X_train, y_train[:,0], X_val, y_val[:,0], X_test, y_test[:,0])
#pred = img_logRegression(X_train_normal, y_train[:,0], X_val_normal, y_val[:,0], X_test_normal, y_test[:,0])
#pred = img_randomForest(X_train, y_train[:,0], X_val, y_val[:,0], X_test, y_test[:,0])
#pred = img_KNN(X_train, y_train[:,0], X_val, y_val[:,0], X_test, y_test[:,0])
#pred = img_baggingClassifier(X_train, y_train[:,0], X_val, y_val[:,0], X_test, y_test[:,0])
#pred = img_boostingClassifier(X_train, y_train[:,0], X_val, y_val[:,0], X_test, y_test[:,0])
#pred = img_MLP(X_train_normal, y_train[:,0], X_val_normal, y_val[:,0], X_test_normal, y_test[:,0])

# =======
# Task A2
# =======
# Pre-process data: Normalisation for logistic regression + MLP
#min_max_scaler = MinMaxScaler()  # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
#X_train_emo_normal = min_max_scaler.fit_transform(X_train_emo)
#X_val_emo_normal = min_max_scaler.transform(X_val_emo)
#X_test_emo_normal = min_max_scaler.transform(X_test_emo)

#pred = img_SVM(X_train_emo, y_train_emo[:,0], X_val_emo, y_val_emo[:,0], X_test_emo, y_test_emo[:,0])
#pred = img_logRegression(X_train_emo_normal, y_train_emo[:,0], X_val_emo_normal, y_val_emo[:,0], X_test_emo_normal, y_test_emo[:,0])
#pred = img_randomForest(X_train_emo, y_train_emo[:,0], X_val_emo, y_val_emo[:,0], X_test_emo, y_test_emo[:,0])
#pred = img_KNN(X_train_emo, y_train_emo[:,0], X_val_emo, y_val_emo[:,0], X_test_emo, y_test_emo[:,0])
#pred = img_baggingClassifier(X_train_emo, y_train_emo[:,0], X_val_emo, y_val_emo[:,0], X_test_emo, y_test_emo[:,0])
#pred = img_boostingClassifier(X_train_emo, y_train_emo[:,0], X_val_emo, y_val_emo[:,0], X_test_emo, y_test_emo[:,0])
#pred = img_MLP(X_train_emo_normal, y_train_emo[:,0], X_val_emo_normal, y_val_emo[:,0], X_test_emo_normal, y_test_emo[:,0])

# =======
# Task B1
# =======
# Pre-process data: Normalisation for logistic regression + MLP
#min_max_scaler = MinMaxScaler()  # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
#X_train_f_normal = min_max_scaler.fit_transform(X_train_f)
#X_val_f_normal = min_max_scaler.transform(X_val_f)
#X_test_f_normal = min_max_scaler.transform(X_test_f)

#pred = img_SVM(X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f)
#pred = img_logRegression(X_train_f_normal, y_train_f, X_val_f_normal, y_val_f, X_test_f_normal, y_test_f)
#pred = img_randomForest(X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f)
#pred = img_KNN(X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f)
#pred = img_baggingClassifier(X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f)
#pred = img_boostingClassifier(X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f)
#pred = img_MLP(X_train_f_normal, y_train_f, X_val_f_normal, y_val_f, X_test_f_normal, y_test_f)

# =======
# Task B2
# =======
#pred = img_CNN(X_train_eye, y_train_eye, X_val_eye, y_val_eye, X_test_eye, y_test_eye)

# ======================================================================================================================
# Learning Curve Plot
'''
fig, axes = lc.plt.subplots(1, 1)
#title = r"Learning Curves (SVM, Linear kernel, C=1.0)"
title = r"Learning Curves (SVM, RBF Kernel, C=50.0, $\gamma=1e-04$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
estimator = svm.SVC(kernel='rbf', C=50, gamma=1e-4)
#estimator = MLPClassifier(max_iter=1e100,alpha=100,solver='lbfgs',
#                          hidden_layer_sizes=(200,200,))
#estimator = LogisticRegression(solver='lbfgs',max_iter=1e50,C=100)
#lc.plot_learning_curve(estimator, title, X_train.reshape((len(X_train), 68*2)), y_train, axes=1, ylim=(0.7, 1.01),
#                    cv=cv, n_jobs=4)
lc.plot_learning_curve(estimator, title, np.concatenate((X_train, X_val)), np.concatenate((y_train[:,0], y_val[:,0])), axes=1, ylim=(0.85, 1.01),
                       cv=cv, n_jobs=4)

lc.plt.show()
'''


# ======================================================================================================================
# ROC Curve Plot
'''
a1_disp = metrics.plot_roc_curve(acc_A1_test, X_test, y_test[:,0], alpha=0.8)
plt.title('Receiver Operating Characteristic (ROC) Curves')
#plt.savefig("roc_A11.png")

ax = plt.gca()
a2_disp = metrics.plot_roc_curve(acc_A2_test, X_test_emo, y_test_emo[:,0], ax=ax, alpha=0.8)
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#
#
#a1_disp.plot(ax=ax)
plt.savefig("roc_A2121.png")
plt.show()
'''