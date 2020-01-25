#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:26:27 2019

@author: chetansrinivasakumar
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cv2 as cv
import scipy
from jeanCV import skinDetector


def argmax2d(X):
    ind = np.unravel_index(np.argmax(np.uint32(X), axis=None), X.shape)
    return ind[0], ind[1]

def findPalmPoint(im):
#    kernel = np.ones((3,3),np.uint8)
#    opening = cv.morphologyEx(im,cv.MORPH_OPEN,kernel, iterations = 3)
    distIm = cv.distanceTransform(np.uint8(im), cv.DIST_C, 5)
#    plt.imshow(distIm)
    return argmax2d(distIm)

def findMassCenter(im):
    handRegion = np.where(im > 0)
    mean = np.mean(handRegion, axis = 1)
    return int(mean[0]), int(mean[1])

def findHandDistanceSignature(palmPoint, mcPoint, contours):
    #Collect contour points.
    border_points = []
    for i in range(len(contours)):
        for j in range(contours[i].shape[0]):
            pt = contours[i][j, :, :]
            border_points.append(pt)

    print("border points shape: ", len(border_points))
    distanceList = []
    thetaList = []
    palmX = palmPoint[0]
    palmY = palmPoint[1]
    mcX = mcPoint[0]
    mcY = mcPoint[1]
    baselineVec = (palmX - mcX, palmY - mcY)
    baselineDist = np.linalg.norm(baselineVec)

    angles = []
    distances = []
    for i in range(len(border_points)):
        border_pt = border_points[i]
        bx = border_pt[0, 0]
        by = border_pt[0, 1]
        borderPointVec = (palmX - bx, palmY - by)
        borderPointDist = np.linalg.norm(borderPointVec)
        ang = np.arccos((baselineVec[0] * borderPointVec[0] + baselineVec[1] * borderPointVec[1]) / (baselineDist * borderPointDist))
        ang = ang * 180.0 / np.pi
        if(ang < 0):
            ang = ang + 360
        distances.append(borderPointDist)
        angles.append(ang)


    n_bins = 40
    bins = np.linspace(0, 360, n_bins)
    hist, bin_edges = np.histogram(a=angles, bins=bins, range=None, normed=None, weights=distances, density=True)

    print(hist)


    return hist

def generate_descriptor(gesture_image):
    print("gesture_image", gesture_image)
#    hand = cv.imread(gesture_image) # Replace hand.jpg with gesture_image
    #print("hand", hand)

    detector = skinDetector(gesture_image)
    detector.find_skin()

    palmY, palmX = findPalmPoint(detector.binary_mask_image)
    print("Palm X: ", palmX, " Palm Y: ", palmY)

    mcY, mcX = findMassCenter(detector.binary_mask_image)
    print("Mc X: ", mcX, " Mc Y: ", mcY)

    im, contours, hierarchy = cv.findContours(detector.binary_mask_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#    cv.drawContours(hand, contours, -1, (255, 0, 0), 3)

    distSig = findHandDistanceSignature((palmX, palmY), (mcX, mcY), contours)
    print("Contour Shape: ", len(contours))
    return distSig

def generate_descriptor_stack(image_folder):
  print(image_folder)
  gesture_types = ["open_hand", "thumbs_up"]#"v", "three"]
  gesture_labels = [0, 1, 2, 3]
  stack_of_descriptors = []
  labels_array = []
  i = 0
  stack_of_descriptors = None
  
  for gesture_type in gesture_types:
    gesture_path = image_folder + gesture_type
    print("gesture_path", gesture_path)
    for gesture_image in glob.glob(gesture_path + '/*.png'):
      distSig = generate_descriptor(gesture_image)
      if(stack_of_descriptors is None):
          stack_of_descriptors = distSig
      else:
          stack_of_descriptors = np.vstack((stack_of_descriptors, distSig))
      labels_array.append(gesture_labels[i])
    i = i + 1
    
  return stack_of_descriptors, np.asarray(labels_array)

if __name__ == "__main__":
  image_folder_path = "Custom_Test/"
  descriptor_training_data, training_labels = generate_descriptor_stack(image_folder_path)
  np.save("descriptor_training_data.npy", descriptor_training_data, allow_pickle=True)
  np.save("descriptor_training_labels.npy", training_labels, allow_pickle=True)
  #old_training_data = np.load("descriptor_training_data.npy")
  #old_labels_data = np.load("descriptor_training_labels.npy")
  #print(old_training_data)
  #print(old_labels_data)

