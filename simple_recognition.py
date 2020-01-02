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
import os
from sklearn.svm import SVC

#-----------------
# TRAINING
#-----------------

# global variables
bg = None
cwd = os.getcwd()
descriptors = np.radom.normal(0.5, 1, size=(12, 12)) #Todo: Store real descriptors
labels = range(12)
sample_images = [cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm", cwd + "/Marcel-Test/A/uniform/A-uniform29.ppm"]

# Generate descriptors for training

# Generate knn model
knn_model = knn_classifcation(descriptors, labels)
accuracies = []

# Generate svm model
svm_model = svm_clssification(descriptors, labels)

def generate_features_and_labels(descriptors, labels):
    #creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    features = le.fit_transform(descriptors)
    encoded_labels = le.fit_transform(labels)
    
    return features, encoded_labels

def knn_classifcation(descriptors, labels):
    features, encoded_labels = generate_features_and_labels(descriptors, labels)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(features,encoded_labels)
    return model

def svm_clssification(descriptors, labels):
    features, encoded_labels = generate_features_and_labels(descriptors, labels)
    clf = SVC(gamma='auto')
    clf.fit(features, encoded_labels)
    return model
    
def prediction(model, test_descriptor):
    predicted= model.predict(test_descriptor)
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
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    
    # random snapshots
    snapshots = np.random.randint(0, 1000, size=20)
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # Todo: change this? convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Todo: Generate test descriptor
        test_descriptor = random.random.normal(0, 1, size=(12, 1))

        # Todo: Make predictions. Test.
        predicted_knn = prediction(knn_model, test_descriptor)
        predicted_svm = np.random.randint(11)
        
        # Todo: Visualize the result
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(clone, "KNN:"predicted_knn + "SVM:" + predicted_svm, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 0, 0), 1, cv2.LINE_AA)
        
        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        
        # save snapshots with their classification
        if num_frames in snapshots:
            accuracies.append(num_frames, predicted)
            cv.imsave(predicted + "_gesture" + num_frames, gray)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()

#-----------------
# VISUALIZATION
#-----------------
