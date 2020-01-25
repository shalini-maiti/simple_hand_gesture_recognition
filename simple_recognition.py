#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:13:36 2019

@author: shalini

"""

#------------------------------------------------------------
# SEGMENT, RECOGNIZE and COUNT fingers from a video sequence
#------------------------------------------------------------

# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import os
from sklearn.svm import SVC
import random
from gesture_recognizer import generate_descriptor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

#-----------------
# TRAINING
#-----------------

# global variables
bg = None
cwd = os.getcwd()
#descriptors = np.random.normal(0.5, 1, size=(12, 12)) #Todo: Store real descriptors
labels = range(12)
sample_images = [cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm"]


def generate_features_and_labels(descriptors, labels):
    #creating labelEncoder
    le = preprocessing.OneHotEncoder()
    print("descriptors SHAPE", descriptors.shape)
    print("LABEL SHAPE", labels.shape)
    # Converting string labels into numbers.
    features = le.fit_transform(descriptors, labels)
    print(features[0])
    encoded_labels = le.fit_transform(labels)
    print("FEATURES SHAPE", features.shape)
    print("LABEL SHAPE", encoded_labels.shape)
    return features, encoded_labels

def knn_classification(descriptors, labels):
    #features, encoded_labels = generate_features_and_labels(descriptors, labels)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(descriptors,labels)
    return model

def svm_classification(descriptors, labels):
    #features, encoded_labels = generate_features_and_labels(descriptors, labels)
    model = SVC(gamma='auto')
    model.fit(descriptors, labels)
    return model

def prediction(model, test_descriptor):
    predicted= model.predict([test_descriptor])
    print(predicted)
    return predicted

#-----------------
# FUNCTIONS FOR TESTING AND ACCURACY
#-----------------

def map_sample_images(label, sample_images):
    return sample_images[label]


#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":

    # Load descriptors and labels for training
    descriptors = np.load('descriptor_training_data.npy')
    descriptors = descriptors
    print("descriptors_shape", descriptors.shape)
    labels = np.load('descriptor_training_labels.npy')
    print("labels_shape", labels.shape)

    # Train and save data
    if os.path.exists("knn_training_weights.sav"):
      # Load knn model
      knn_model = pickle.load(open("knn_training_weights.sav", 'rb'))
    else:
      # Generate knn model
      knn_model = knn_classification(descriptors, labels)
      pickle.dump(knn_model, open("knn_training_weights.sav", 'wb'))

    accuracies_knn = []

    if os.path.exists("svm_training_weights.sav"):
      #Load svm model
      svm_model = pickle.load(open("svm_training_weights.sav", 'rb'))
    else:
      # Generate svm model
      svm_model = svm_classification(descriptors, labels)
      pickle.dump(svm_model, open("svm_training_weights.sav", 'wb'))
    accuracies_svm = []

    # initialize accumulated weight
    accumWeight = 0.5
    
    testIm = cv2.imread("./Custom_Test/random_test/test_oh0.jpg")
    testIm_descriptor = generate_descriptor("./Custom_Test/random_test/test_oh0.jpg")
    testIm_descriptor = np.asarray(testIm_descriptor)
    testIm_prediction = prediction(knn_model, testIm_descriptor)
    if(testIm_prediction == 0):
        print("OPEN HAND!")
    else:
        print("THUMBS UP!")
   
#    # get the reference to the webcam
#    camera = cv2.VideoCapture(0)
#
#    # region of interest (ROI) coordinates
#    top, right, bottom, left = 10, 350, 225, 590
#
#    # initialize num of frames
#    num_frames = 0
#
#    # random snapshots
#    snapshots = np.random.randint(0, 100, size=20)
#    # keep looping, until interrupted
#    while(True):
#        # get the current frame
#        grabbed, frame = camera.read()
#
#        # resize the frame
#        frame = imutils.resize(frame, width=700)
#
#        # flip the frame so that it is not the mirror view
#        frame = cv2.flip(frame, 1)
#
#        # clone the frame
#        clone = frame.copy()
#
#        # get the height and width of the frame
#        (height, width) = frame.shape[:2]
#
#        # get the ROI
#        roi = frame[top:bottom, right:left]
#
#        # Todo: change this? convert the roi to grayscale and blur it
#        #gray = clone
#        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#        #gray = cv2.GaussianBlur(gray, (7, 7), 0)
#
#        # Todo: Generate test descriptor
#        #plt.imshow(gray)
#        cv2.imwrite('image_temp.jpg', frame)
#        test_descriptor = generate_descriptor('image_temp.jpg')
#        print("test_descriptor", test_descriptor.shape)
#        # Todo: Make predictions. Test.
#        predicted_knn = prediction(knn_model, test_descriptor)
#        predicted_svm = prediction(svm_model, test_descriptor)
#
#        # Todo: Visualize the result
#        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
##        cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
#        cv2.putText(clone, "KNN:" + str(predicted_knn) + "SVM:" + str(predicted_svm), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 0, 0), 1, cv2.LINE_AA)
#
#        # increment the number of frames
#        num_frames += 1
#
#        # display the frame with segmented hand
#        cv2.imshow("Video Feed" + str(num_frames), clone)
#
#
#        # save snapshots with their classification
#        if num_frames in snapshots:
#            accuracies_knn.append([num_frames, predicted_knn])
#            cv2.imwrite('snapshots/'+ str(num_frames) + '.jpg', clone)
#            accuracies_svm.append([num_frames, predicted_svm])
#            #cv2.imwrite('snapshots/'+ predicted_svm + "_gesture" + num_frame + '.jpg', clone)
#
#        # observe the keypress by the user
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
#        print(num_frames)
## free up memory
#camera.release()
#cv2.destroyAllWindows()
#np.save("accuracies_knn", accuracies_knn) # Manually annotate
#np.save("accuracies_svm", accuracies_svm) # Manually annotate
#print("Fin")
#
##-----------------
## VISUALIZATION
##-----------------
