import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
from numpy import save
from canny_edge_detection import Canny_detector

# PATH TO ALL IMAGES
# Task A
global basedir, image_paths, target_size
basedir = './Datasets/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

# Task B
basedir_taskB = './Datasets/cartoon_set'
images_dir_taskB = os.path.join(basedir_taskB,'img')
labels_filename_taskB = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#jawline_points = list(range(0, 17))
#right_eyebrow_points = list(range(17, 22))
#left_eyebrow_points = list(range(22, 27))
#nose_points = list(range(27, 36))
#right_eye_points = list(range(36, 42))
#left_eye_points = list(range(42, 48))
#mouth_outline_points = list(range(48, 61))
#mouth_inner_points = list(range(61, 68))

# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def extract_features_labels_task_a():
    """
    This funtion extracts the landmarks features for all images in the folder 'Datasets/celeba/img'.
    It also extracts the gender label and emotion label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
        emotion_labels:     an array containing the emotion label (not smiling=0 and smiling=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    emotion_labels = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_gender_labels = []
        all_emotion_labels = []
        fail = []
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is None:
                fail.append(file_name)
                print(file_name)
            if features is not None:
                all_features.append(features)
                all_gender_labels.append(gender_labels[file_name])
                all_emotion_labels.append(emotion_labels[file_name])

        print(fail.sort())

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_gender_labels) + 1)/2  # simply converts the -1 into 0, so male=0 and female=1
    emotion_labels = (np.array(all_emotion_labels) + 1)/2  # simply converts the -1 into 0, so not smiling=0 and smiling=1

    # save to npy file
    #save('landmark_features.npy', landmark_features)
    #save('gender_labels.npy', gender_labels)
    #save('emotion_labels.npy', emotion_labels)

    return landmark_features, gender_labels, emotion_labels


def extract_features_labels_task_b():
    """
    This funtion extracts the landmarks features for all images in the folder 'Datasets/cartoon_set/img'.
    It also extracts the face shape label and eye color label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        face_shape_labels:  an array containing the face shape label for each image in which a face was detected
        eye_color_labels:   an array containing the eye color label for each image in which a face was detected
    """
    image_paths = [os.path.join(images_dir_taskB, l) for l in os.listdir(images_dir_taskB)]
    target_size = None
    labels_file = open(os.path.join(basedir_taskB, labels_filename_taskB), 'r')
    lines = labels_file.readlines()
    face_shape_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    eye_color_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    if os.path.isdir(images_dir_taskB):
        all_features = []
        all_features_eye = []
        all_face_shape_labels = []
        all_eye_color_labels = []
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('/')[-1]
            # load image
            img = image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic')


            cropped_im = img.crop((160, (160+425)/2, 340, 425))  # left, top, right, bottom
            cropped_im = np.asarray(cropped_im)  # numpy array
            #features = cv2.Canny(cropped_im)  # Canny image
            features = Canny_detector(cropped_im)  # Canny image
            cropped_im_eye = img.crop((190, 250, 220, 270))  # left, top, right, bottom
            features_eye = np.asarray(cropped_im_eye)  # numpy array
            if features is not None:
                all_features.append(features)
                all_face_shape_labels.append(face_shape_labels[file_name])
            if features_eye is not None:
                all_features_eye.append(features_eye)
                all_eye_color_labels.append(eye_color_labels[file_name])

    jawline_landmark_features = np.array(all_features)
    #jawline_landmark_features2 = all_features2[:,np.array(jawline_points)]
    #eye_landmark_features = landmark_features[:,np.array(right_eye_points + left_eye_points)]
    eye_landmark_features = np.array(all_features_eye)
    face_shape_labels = np.array(all_face_shape_labels)
    eye_color_labels = np.array(all_eye_color_labels)

    # save to npy file
    #save('jawline_canny_features.npy', jawline_landmark_features)
    #save('eye_region_features.npy', eye_landmark_features)
    #save('face_shape_labels.npy', face_shape_labels)
    #save('eye_color_labels.npy', eye_color_labels)

    return jawline_landmark_features,eye_landmark_features,face_shape_labels,eye_color_labels

