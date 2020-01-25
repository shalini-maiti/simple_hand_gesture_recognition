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

width_ = 0
height_ = 0

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

def centerCoord(pt, w, h):
    if(isinstance(pt, (np.ndarray, np.matrix))):
        sh = np.zeros((1, 2))
        sh[0, 0] = pt[0, 0] - (w/2)
        sh[0, 1] = (h/2) - pt[0, 1]
        return sh
    
    ptnew = []
    ptnew.append((pt[0] - (w/2)))
    ptnew.append(((h/2) - pt[1]))
    return ptnew

def findHandDistanceSignature(palmPoint, mcPoint, contours):
    #Collect contour points.
#    max_idx = 0
#    for i in range(len(contours) - 1):
#        if(contours[i].shape[0] < contours[i + 1].shape[0]):
#            max_idx = i + 1
#            
    border_points = []
    for i in range(len(contours)):
        for j in range(contours[i].shape[0]):
            pt = contours[i][j, :, :]
            pt = centerCoord(pt, width_, height_)
            border_points.append(pt)
    
    palmPoint = centerCoord(palmPoint, width_, height_)
    mcPoint = centerCoord(mcPoint, width_, height_)
    #Shift coordinates to image center, so we don't have problems with reflex angles in 
    #descriptor computation. 
    print("border points shape: ", len(border_points))
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
    hand = cv.imread(gesture_image) # Replace hand.jpg with gesture_image
    global width_
    global height_
    width_ = hand.shape[1]
    height_ = hand.shape[0]
    # print("hand shape", hand.shape)

    detector = skinDetector(gesture_image)
    detector.find_skin()
    
    palmY, palmX = findPalmPoint(detector.binary_mask_image)
    print("Palm X: ", palmX, " Palm Y: ", palmY)
    mcY, mcX = findMassCenter(detector.binary_mask_image)
    print("Mc X: ", mcX, " Mc Y: ", mcY)
    
#    pp = []
#    pp.append(palmX)
#    pp.append(palmY)
#    detector.refine_mask(pp)
#    
#    palmY, palmX = findPalmPoint(detector.binary_mask_image)

    thresh_mask = detector.binary_mask_image.astype(np.uint8)
    im, contours, hierarchy = cv.findContours(detector.binary_mask_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(hand, contours, -1, (255, 0, 0), 3)
    print("Haha")

    distSig = findHandDistanceSignature((palmX, palmY), (mcX, mcY), contours)
    
    #Plot results. 
#    f = plt.figure()
# #    f.add_subplot(1, 2, 1)
#     handPlt = cv.drawMarker(hand, (palmX, palmY), (255, 0, 0), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2, line_type=cv.LINE_AA)
#     handPlt = cv.drawMarker(handPlt, (mcX, mcY), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2, line_type=cv.LINE_AA)
#    plt.imshow(handPlt)
#    f.add_subplot(1, 2, 2)
#    plt.imshow(detector.binary_mask_image)
    
    print("haha1")
    return distSig, None, thresh_mask

def generate_descriptor_stack(image_folder):
  print(image_folder)
#  gesture_types = ["open_hand", "thumbs_up", "v", "three"]
  gesture_types = ["open_hand", "thumbs_up"]
  stack_of_descriptors = None
  gesture_labels = [0, 1, 2, 3]
  labels_array = []
  i = 0
  for gesture_type in gesture_types:
    label = gesture_labels[i]
    gesture_path = image_folder + gesture_type + "/*.png"
    
    count = 0
    gesture_im_list = []
    gesture_im_list = sorted(glob.glob(gesture_path))
    for gesture_image in gesture_im_list:
        print("gesture image path", gesture_image)
        if(count < len(gesture_im_list)):
          distSig, pltImage, maskImage = generate_descriptor(gesture_image)
          
#          image_name = gesture_image.split(sep="/")
#          image_name = image_name[len(image_name) - 1]
#          contour_img_path = image_folder + "/contour/" + image_name 
#          mask_img_path = image_folder + "/mask/" + image_name
#          cv.imwrite(contour_img_path, pltImage)
#          cv.imwrite(mask_img_path, maskImage)
          
          if(stack_of_descriptors is None):
            stack_of_descriptors = distSig

          else:
            stack_of_descriptors = np.vstack((stack_of_descriptors, distSig))

          labels_array.append(label)
#          print("DEscriptor sHape", np.array(stack_of_descriptors).shape)
          count = count + 1
        
        else:
            break;
    i = i + 1
           
  return stack_of_descriptors, labels_array

if __name__ == "__main__":
  image_folder_path = "./Custom_Test/"
  descriptor_training_data, training_labels = generate_descriptor_stack(image_folder_path)
  np.save("descriptor_training_data.npy", descriptor_training_data)
  np.save("descriptor_training_labels.npy", np.asarray(training_labels))
  #old_training_data = np.load("descriptor_training_data.npy")
  #old_labels_data = np.load("descriptor_training_labels.npy")
  #print(old_training_data)
  #print(old_labels_data)

