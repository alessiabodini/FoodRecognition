from resnet import FeaturesExtractor
from pathlib import Path
import numpy as np
import csv
import cv2
import os

path = Path(os.path.join('C:/', 'Users', 'ale19', 'Downloads', 'Food-101')) # your path of the dataset
path_img = path/'images'
n_classes = 101
max_img = 999

def loadImages(t_set, n_images_per_class, phase):
    
    images, labels = [], []
    n_class = 0
    idx = 0
    
    for c in t_set:
        print(c)
        for i in range(n_images_per_class):
            idx += 1
            name = t_set[c][i]
            labels.append(n_class)
            img = cv2.imread(os.path.join(path_img, name))
            img = cv2.resize(img,(64,64))
            images.append(img)
        n_class += 1

    return np.array(images), np.array(labels)


def extractFeatures(t_set, n_images_per_class, phase):

    # Extractor initialization
    extractor = FeaturesExtractor()

    # Load images
    print('Loading images...')
    images, labels = loadImages(t_set, n_images_per_class, phase)

    # Infer from all images 
    print('Extracting features...')
    n_features = 2048
    n_images = images.shape[0]
    features = np.zeros((n_images, n_features))
    for i in range(n_images):
        if i % 100 == 0:
            print(i)
        img = images[i]
        features[i] = extractor.getFeatures(img)        
    
    return np.array(labels), features


def extractLabels(t_set, n_images_per_class):
    labels = []
    n_class = 0
    
    for c in range(n_classes):
        for i in range(n_images_per_class):
            labels.append(n_class)
        n_class += 1

    return np.array(labels)
