from resnet import FeaturesExtractor
from pathlib import Path
import numpy as np
import csv
import cv2
import os

path = Path(os.path.join('C:/', 'Users', 'ale19', 'Downloads', 'Food-101'))
path_h5 = path
path_img = path/'images'
path_meta = path/'meta/meta'
path_working = '/kaggle/working/'

<<<<<<< HEAD

def extractFeatures(filename):
=======
def extractFeatures(t_set, n_images, n_images_per_class):
>>>>>>> parent of d3718eb... Importing from .h5 files

    # Extractor initialization
    extractor = FeaturesExtractor()

<<<<<<< HEAD
    # Import information from file .h5
    f = h5py.File(filename, 'r')
    category = f['category']
    category_names = f['category_names']
    images = f['images']
    n_images = images.shape[0]

=======
>>>>>>> parent of d3718eb... Importing from .h5 files
    # Infer from all images (or until one is reached)
    n_features = 2048
    labels = []
    features = np.zeros((n_images, n_features))
<<<<<<< HEAD
    for i in range(n_images):
        if i % 100 == 0:
            print(i)
        labels.append(category_names[category[i] == True][0].decode('utf-8'))
        img = images[i]

        #feature_matrix = np.zeros((64,64)) 
        #for i in range(img.shape[0]):
            #for j in range(img.shape[1]):
                #feature_matrix[i][j] = ((int(img[i,j,0]) + int(img[i,j,1]) + int(img[i,j,2]))/3)
        #features = np.reshape(feature_matrix, (64*64)) 
        features[i] = extractor.getFeatures(img)
    
    return np.array(labels, dtype='str'), features


def extractLabels(filename):
    # Import information from file .h5
    f = h5py.File(filename, 'r')
    category = f['category']
    category_names = f['category_names']
    n_images = category.shape[0]

    labels = []
    for i in range(n_images):
        if i % 100 == 0:
            print(i)
        labels.append(category_names[category[i] == True][0].decode('utf-8'))
    
    return np.array(labels, dtype='str')
=======
    idx = -1
    for c in t_set:
        print(c)
        img_per_set = 0
        for name in t_set[c]:
            print(img_per_set, end=' ')
            idx += 1
            img_per_set += 1
            labels.append(name)
            img = cv2.imread(os.path.join(path_img, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(128,128))
            # img_list size has to be (N,W,H,C), output has size (N,2048)
            features[idx] = extractor.getFeatures(img)
            if img_per_set == n_images_per_class:
                break
        print()
        if c == 'bruschetta': # ONLY THE FIRST 11 CLASSES
            break

    #print(features[:5,:])
            
    return np.array(labels), features


def extractLabels(t_set, n_images, n_images_per_class):
    labels = []
    for c in t_set:
        img_per_set = 0
        for name in t_set[c]:
            img_per_set += 1
            labels.append(c)
            if img_per_set == n_images_per_class:
                break
    return np.array(labels)
>>>>>>> parent of d3718eb... Importing from .h5 files
