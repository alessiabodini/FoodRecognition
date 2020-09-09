import os
import csv
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from knn import knn
from svm import svm
from nnet import nnet
from pathlib import Path
from tools import extractFeatures, extractLabels, loadImages


path = Path(os.path.join('C:/', 'Users', 'ale19', 'Downloads', 'Food-101')) # your path of the dataset
path_h5 = path
path_img = path/'images'
path_meta = path/'meta/meta'
path_working = '/kaggle/working/'

start_time = time.time()

# Importing the dataset ---------------------------------------------------------------------------------------

# Map from folder to space seperated labels
def label_from_folder_map(class_to_label_map):
    return lambda o: class_to_label_map[(o.parts if isinstance(o, Path) else o.split(os.path.sep))[-2]]

# Develop dictionary mapping from classes to labels
classes = pd.read_csv(path_meta/'classes.txt', header=None, index_col=0)
classes_list = classes.index.tolist()
labels = pd.read_csv(path_meta/'labels.txt', header=None)
classes['map'] = labels[0].values
classes_to_labels_map = classes['map'].to_dict()
label_from_folder_food_func = label_from_folder_map(classes_to_labels_map)

# Setup the training set of images
train_df = pd.read_csv(path_meta/'train.txt', header=None).apply(lambda x : x + '.jpg')
train_set = dict((c, []) for c in classes_list)
for i in range(len(train_df)):
    train_set[Path(train_df[0][i]).parts[-2]].append(train_df[0][i])

# Setup the testing set of images
test_df = pd.read_csv(path_meta/'test.txt', header=None).apply(lambda x : x + '.jpg')
test_set = dict((c, []) for c in classes_list)
for i in range(len(test_df)):
    test_set[Path(test_df[0][i]).parts[-2]].append(train_df[0][i])
    
# -----------------------------------------------------------------------------------------------------

# For KNN and SVM >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Detect the features for train and test set ----------------------------------------------------------

n_classes = 10
n_features = 2048
train_file_feat = 'train_features_100x10.csv'
test_file_feat =  'test_features_10x10.csv'
train_file_labels = 'train_labels_100x10.csv'
test_file_labels = 'test_labels_10x10.csv'

n_images_per_class = 100
if os.path.isfile(train_file_feat):
    train_features = np.genfromtxt(train_file_feat, delimiter=',')
else:
    train_labels, train_features = extractFeatures(train_set, n_images_per_class)
    with open(train_file_feat, 'w', newline='') as filename:
        writer = csv.writer(filename)
        writer.writerows(train_features)    
if os.path.isfile(train_file_labels):
    train_labels = np.genfromtxt(train_file_labels, delimiter=',')
else:
    if 'train_labels' not in globals():
        train_labels = extractLabels(train_set, n_images_per_class)
    with open(train_file_labels, 'w', newline='') as filename:
        writer = csv.writer(filename, delimiter=',')
        writer.writerow(train_labels)

n_images_per_class = 10
if os.path.isfile(test_file_feat):
    test_features = np.genfromtxt(test_file_feat, delimiter=',')
else:
    test_labels, test_features = extractFeatures(test_set, n_images_per_class)
    with open(test_file_feat, 'w', newline='') as filename:
        writer = csv.writer(filename)
        writer.writerows(test_features)    
if os.path.isfile(test_file_labels):
    test_labels = np.genfromtxt(test_file_labels, delimiter=',')
else:
    if 'test_labels' not in globals():
        test_labels = extractLabels(test_set, n_images_per_class)
    with open(test_file_labels, 'w', newline='') as filename:
        writer = csv.writer(filename, delimiter=',')
        writer.writerow(test_labels)

print('Import completed.\n')

# -----------------------------------------------------------------------------------------------------

# Classification process ------------------------------------------------------------------------------

print('Classification with KNN...')
predicted_knn = knn(train_features, test_features, train_labels, test_labels)
print('Classification with SVM...')
predicted_svm = svm(train_features, test_features, train_labels, test_labels)

# -----------------------------------------------------------------------------------------------------

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# For NN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Import images for train and test set ----------------------------------------------------------------

n_images_per_class = 100
train_images, train_labels = loadImages(train_set, n_images_per_class)
n_images_per_class = 10
test_images, test_labels = loadImages(test_set, n_images_per_class)

# -----------------------------------------------------------------------------------------------------

# Classification process ------------------------------------------------------------------------------

print('Classification with NN...')
predicted_nn = nnet(train_images, test_images, train_labels, test_labels, classes_list)

# -----------------------------------------------------------------------------------------------------

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Showing results images and predictions --------------------------------------------------------------

random.seed()
images_to_print = [random.randint(0,n_images_per_class*n_classes-1) for i in range(4)]
print('Predicted by KNN: ', ' '.join('%20s' % classes_list[int(predicted_knn[i])] for i in images_to_print))
print('Predicted by SVM: ', ' '.join('%20s' % classes_list[int(predicted_svm[i])] for i in images_to_print))
print('Predicted by NN:  ', ' '.join('%20s' % classes_list[int(predicted_nn[i])] for i in images_to_print))
print('Ground Truth:     ', ' '.join('%20s' % classes_list[int(test_labels[i])] for i in images_to_print))

fig = plt.figure(figsize=(8, 8))
for i in range(len(images_to_print)):
    fig.add_subplot(1, len(images_to_print), i+1)
    img = test_images[images_to_print[i]]
    plt.imshow(img)
plt.show()

# -----------------------------------------------------------------------------------------------------

total_time = time.time() - start_time
print('\nProcessing time: %d min.' % int(total_time/60) if total_time >= 60 
        else '\nProcessing time: %d sec.' % int(total_time))
