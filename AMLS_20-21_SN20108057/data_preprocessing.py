from sklearn.model_selection import train_test_split
import face_landmarks as lm
from sklearn.preprocessing import OneHotEncoder
from numpy import load  # load numpy array from npy file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data preprocessing
def data_preprocessing_task_a():
    # Task A dataset
    X,y,y_emotion = lm.extract_features_labels_task_a()

    # Load data
    #X = load('landmark_features.npy')
    #y = load('gender_labels.npy')
    #y_emotion = load('emotion_labels.npy')

    # One hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    y_integer_encoded = y.reshape(len(y), 1)
    y = onehot_encoder.fit_transform(y_integer_encoded)

    y_emotion_integer_encoded = y_emotion.reshape(len(y_emotion), 1)
    y_emotion = onehot_encoder.fit_transform(y_emotion_integer_encoded)

    X = X.reshape((len(X), 68*2))

    # Create test and train sets from one dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train_emo, X_test_emo, y_train_emo, y_test_emo = train_test_split(X, y_emotion, test_size=0.2, random_state=1)
    # Create a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    X_train_emo, X_val_emo, y_train_emo, y_val_emo = train_test_split(X_train_emo, y_train_emo, test_size=0.25, random_state=1)

    return X_train, y_train, X_val, y_val, X_test, y_test, \
           X_train_emo, y_train_emo, X_val_emo, y_val_emo, X_test_emo, y_test_emo


def data_preprocessing_task_b():
    # Task B dataset
    X_face_shape, X_eye_color, y_face_shape, y_eye_color = lm.extract_features_labels_task_b()

    # Load data
    #X_face_shape = load('jawline_canny_features.npy')
    #y_face_shape = load('face_shape_labels.npy')
    #X_eye_color = load('eye_region_features.npy')
    #y_eye_color = load('eye_color_labels.npy')

    # Task B2: One hot encoding for CNN
    onehot_encoder = OneHotEncoder(sparse=False)
    y_integer_encoded = y_eye_color.reshape(len(y_eye_color), 1)
    y_eye_color = onehot_encoder.fit_transform(y_integer_encoded)

    # Create test and train sets from one dataset
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_face_shape, y_face_shape, test_size=0.2, random_state=1)
    X_train_eye, X_test_eye, y_train_eye, y_test_eye = train_test_split(X_eye_color, y_eye_color, test_size=0.2, random_state=1)

    # Create a validation set
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(X_train_f, y_train_f, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    X_train_eye, X_val_eye, y_train_eye, y_val_eye = train_test_split(X_train_eye, y_train_eye, test_size=0.25, random_state=1)

    # Task B1: Standardisation before PCA
    scaler = StandardScaler()
    # Fit on training set only
    scaler.fit(X_train_f.reshape((len(y_train_f),133*180)))
    # Apply transform to the training set, the validation set and the test set.
    X_train_f = scaler.transform(X_train_f.reshape((len(y_train_f),133*180)))
    X_val_f = scaler.transform(X_val_f.reshape((len(y_val_f),133*180)))
    X_test_f = scaler.transform(X_test_f.reshape((len(y_test_f),133*180)))

    # Task B1: PCA
    pca = PCA(n_components=50, random_state=1) # Number of components to keep
    # Fit PCA on the training set only, and apply the mapping (transform) to the training set
    X_train_f = pca.fit_transform(X_train_f)
    # Apply the mapping (transform) to the validation set and the test set.
    X_val_f = pca.transform(X_val_f)
    X_test_f = pca.transform(X_test_f)

    return X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f, \
           X_train_eye, y_train_eye, X_val_eye, y_val_eye, X_test_eye, y_test_eye