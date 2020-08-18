from resnet import FeaturesExtractor
from pathlib import Path
import numpy as np
import h5py
import csv
import cv2
import os


def loadImages(filename):

    # Import information from file .h5
    f = h5py.File(filename, 'r')
    category = f['category']
    category_names = f['category_names']
    images = f['images']

    return images, category


def extractFeatures(filename):

    # Extractor initialization
    extractor = FeaturesExtractor()

    # Load imagesù
    images, category = loadImages(filename)

    # Infer from all images (or until one is reached)
    n_features = 2048
    n_images = images.shape[0]
    labels = []
    features = np.zeros((n_images, n_features))
    for i in range(n_images):
        if i % 100 == 0:
            print(i)
        img = images[i]

        labels.append([j for j in range(category.shape[1]) if category[i][j] == True][0])

        #feature_matrix = np.zeros((64,64)) 
        #for i in range(img.shape[0]):
            #for j in range(img.shape[1]):
                #feature_matrix[i][j] = ((int(img[i,j,0]) + int(img[i,j,1]) + int(img[i,j,2]))/3)
        #features = np.reshape(feature_matrix, (64*64)) 
        features[i] = extractor.getFeatures(img)
    
    return np.array(labels), features


def extractLabels(filename):
    # Load images
    _, category = loadImages(filename)

    labels = []
    for i in range(category.shape[0]):
        if i % 100 == 0:
            print(i)
        labels.append([j for j in range(category.shape[1]) if category[i][j] == True][0])

    return np.array(labels)

