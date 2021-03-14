from result_display import display_metric_results_A
from result_display import display_metric_results_B
from sklearn import svm
from hypopt import GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


# ======================================================================================================================
# Bagging Classification Model
def img_baggingClassifier(training_images, training_labels, val_images, val_labels,test_images, test_labels): # k: the number of decision trees in the ensemble
    #Create KNN object with a K coefficient
    param_grid = {'n_estimators': [128,256,512,1024,2048],
                  'max_samples': [0.5,0.75,1],
                  'max_features': [0.1,0.25,0.5,0.75,1]}
    gs = GridSearch(model=BaggingClassifier(), param_grid=param_grid)
    clf = gs.fit(training_images, training_labels, val_images, val_labels)
    print("Best estimator found by grid search:")
    print(clf)
    training_pred = clf.predict(training_images)
    val_pred = clf.predict(val_images)
    test_pred = clf.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    #display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# Boosting Classification Model
def img_boostingClassifier(training_images, training_labels, val_images, val_labels,test_images, test_labels): # k: the maximum number of DecisionTreeClassifier(max_depth=1) at which boosting is terminated
    # AdaBoost takes Decision Tree as its base-estimator model by default.
    param_grid = {'n_estimators': [1,2,4,8,16,32,64,128,256,512,1024,2048,4096],
                  'algorithm': ['SAMME','SAMME.R'],}
    gs = GridSearch(model=AdaBoostClassifier(), param_grid=param_grid)
    clf = gs.fit(training_images, training_labels, val_images, val_labels) # # Fit KNN model # Build a boosted classifier from the training set
    print("Best estimator found by grid search:")
    print(clf)
    training_pred = clf.predict(training_images)
    val_pred = clf.predict(val_images)
    test_pred = clf.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    #display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# SVM Classification Model
def img_SVM(training_images, training_labels, val_images, val_labels,test_images, test_labels):
    #clf = svm.SVC(kernel='linear')
    #clf.fit(training_images, training_labels)
    print("Fitting the classifier to the training set")

    param_grid = {'C': [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,5e1,1e2,5e2,1e3,5000],
                  'gamma': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4], } #
    #param_grid = {'C': [1e1,5e1,1e2,5e2,1e3,5e3],
    #              'degree': [3,4,5,6], }
    #param_grid = {'C': [0.001,0.01,0.1,0.5,1,10,5,50,100,500,1000],}
    gs = GridSearch(model=svm.SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid)
    #gs = GridSearch(model=svm.SVC(kernel='poly', class_weight='balanced'), param_grid=param_grid)
    #gs = GridSearch(model=svm.SVC(kernel='linear', class_weight='balanced'), param_grid=param_grid)
    clf = gs.fit(training_images, training_labels, val_images, val_labels)

    print("Best estimator found by grid search:")
    print(clf) # equivalent to print(gs.best_estimator_)

    training_pred = clf.predict(training_images) # equavalent to gs.best_estimator_.predict(training_images)
    val_pred = clf.predict(val_images)
    test_pred = clf.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    #display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# KNN Classification Model
def img_KNN(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    #Create KNN object with a K coefficient
    param_grid = {'n_neighbors': list(range(1,50)),
                  'leaf_size': list(range(1,50)),
                  'p': [1,2]}
    gs = GridSearch(model=KNeighborsClassifier(), param_grid=param_grid)
    # Train the model using the training sets
    neigh = gs.fit(training_images, training_labels, val_images, val_labels) # Fit KNN model
    print("Best estimator found by grid search:")
    print(neigh)
    training_pred = neigh.predict(training_images)
    val_pred = neigh.predict(val_images)
    test_pred = neigh.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    #display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# Logistic Regression Classification Model
def img_logRegression(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    # Build Logistic Regression Model
    param_grid = {'C': [1,5,10,50,100,500,1000,5000],}
    gs = GridSearch(model=LogisticRegression(solver='lbfgs',max_iter=1e50), param_grid=param_grid)
    #logreg = LogisticRegression(solver='lbfgs',max_iter=1e50,C=100)
    # Train the model using the training sets
    logreg = gs.fit(training_images, training_labels, val_images, val_labels)
    #logreg.fit(training_images, training_labels)
    print("Best estimator found by grid search:")
    print(logreg)
    #logreg.fit(training_images, training_labels)
    training_pred = logreg.predict(training_images)
    val_pred = logreg.predict(val_images)
    test_pred = logreg.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    #display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# MLP Classification Model
def img_MLP(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    param_grid = {'alpha': [1e-4,1e-3,1e-2,0.1,1,100,1000], # 1e-4,1e-3,1e-2,0.1,1,100,1000
                  'solver': ['lbfgs','adam','sgd'], # 'lbfgs','adam','sgd'
                  'hidden_layer_sizes': [(200,200,),(500,)], # (200,200,),(500,)
                  'learning_rate_init': [0.00001,0.0001,0.001,0.005,0.01,0.05] # 0.00001,0.0001,0.001,0.005,0.01,0.05
                  }

    gs = GridSearch(model=MLPClassifier(max_iter=1e100), param_grid=param_grid)
    print("finish grid search")
    clf = gs.fit(training_images, training_labels, val_images, val_labels)
    print("Best estimator found by grid search:")
    print(clf)

    training_pred = clf.predict(training_images)
    val_pred = clf.predict(val_images)
    test_pred = clf.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    #display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# Random Forest Classification Model
def img_randomForest(training_images, training_labels, val_images, val_labels,test_images, test_labels):
    param_grid = {'n_estimators': [600,1000,1200,1400,800],  # Number of trees in random forest
                  # Number of features to consider at every split
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  # Method of selecting samples for training each tree
                  'bootstrap': [True,False],
                  }
    gs = GridSearch(model=RandomForestClassifier(), param_grid=param_grid)
    clf = gs.fit(training_images, training_labels, val_images, val_labels)
    print("Best estimator found by grid search:")
    print(clf)
    training_pred = clf.predict(training_images)
    val_pred = clf.predict(val_images)
    test_pred = clf.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    #display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# CNN Classification Model
def img_CNN(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    # Normalize the images
    training_images = (training_images/255)-0.5
    val_images = (val_images/255)-0.5
    test_images = (test_images/255)-0.5

    # Create a `Sequential` model
    model = Sequential([
        Conv2D(32,kernel_size=(5,5),strides=(1,1),  # num_filter, filter_size
               activation='relu',
               input_shape=(20,30,3)),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        Dropout(0.25),
        Conv2D(64,(5,5),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(1000,activation='relu'),
        Dropout(0.5),
        Dense(5,activation='softmax'),
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    #to_categorical(training_labels,num_classes=5)
    callback = EarlyStopping(monitor='loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    clf = model.fit(training_images, training_labels,
                    batch_size=128,  # 16,32,64,128
                    epochs=100,
                    verbose=1,
                    validation_data=(val_images, val_labels),
                    callbacks=[callback])

    training_score = model.evaluate(training_images, training_labels, verbose=0)
    validation_score = model.evaluate(val_images, val_labels, verbose=0)
    test_score = model.evaluate(test_images, test_labels, verbose=0)

    print(f"{'Number of epochs for training:':<31}{len(clf.history['loss']):>10.0f}")
    print(f"{'Training Set Accuracy Score:':<31}{training_score[1]:>10.4f}")
    print(f"{'Validation Set Accuracy Score:':<31}{validation_score[1]:>10.4f}")
    print(f"{'Test Set Accuracy Score:':<31}{test_score[1]:>10.4f}")

    plt.grid()
    plt.plot(clf.history['acc'])
    plt.plot(clf.history['val_acc'])
    plt.title("Model Accuracy (CNN, 'Adam' Optimizer, batch_size=128)")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training score', 'Cross-validation score'], loc='best')
    plt.savefig("B2_128_epoc20.png")
    plt.show()
    return training_labels[1]
