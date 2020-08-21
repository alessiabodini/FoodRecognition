import os
import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tools import extractFeatures, extractLabels
from knn import knn
from svm import svm
from nnet import nnet
from nnet_feat import nnet_feat

path = Path(os.path.join('C:/', 'Users', 'ale19', 'Downloads', 'Food-101'))
path_h5 = path
path_img = path/'images'
path_meta = path/'meta/meta'
path_working = '/kaggle/working/'

start_time = time.time()

# Importing the dataset ---------------------------------------------------------------------------------------

# Modify the from folder function in fast.ai to use the dictionary mapping from folder to space seperated labels
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
    
#img = Image.open(os.path.join(path_img, test_set['apple_pie'][1]))
#plt.imshow(np.array(img))

# -------------------------------------------------------------------------------------------------------------

# Detect the features for train and test set ------------------------------------------------------------------

n_features = 2048
train_file = os.path.join(path_h5, 'food_c101_n10099_r64x64x3.h5')
test_file = os.path.join(path_h5, 'food_test_c101_n1000_r64x64x3.h5')
train_file_feat = 'train_features.csv'
test_file_feat =  'test_features.csv'
train_file_labels = 'train_labels.csv'
test_file_labels = 'test_labels.csv'
train_labels, test_labels = [], []

if os.path.isfile(train_file_feat):
    train_features = np.genfromtxt(train_file_feat, delimiter=',')
else:
    train_labels, train_features = extractFeatures(train_file)
    with open(train_file_feat, 'w', newline='') as filename:
        writer = csv.writer(filename)
        writer.writerows(train_features)    
if os.path.isfile(train_file_labels):
    train_labels = np.genfromtxt(train_file_labels, delimiter=',')
else:
    if train_labels == []:
        train_labels = extractLabels(train_file)
    with open(train_file_labels, 'w', newline='') as filename:
        writer = csv.writer(filename, delimiter=',')
        writer.writerow(train_labels)

if os.path.isfile(test_file_feat):
    test_features = np.genfromtxt(test_file_feat, delimiter=',')
else:
    test_labels, test_features = extractFeatures(test_file)
    with open(test_file_feat, 'w', newline='') as filename:
        writer = csv.writer(filename)
        writer.writerows(test_features)    
if os.path.isfile(test_file_labels):
    test_labels = np.genfromtxt(test_file_labels, delimiter=',')
else:
    if test_labels == []:
        test_labels = extractLabels(test_file)
    with open(test_file_labels, 'w', newline='') as filename:
        writer = csv.writer(filename, delimiter=',')
        writer.writerow(test_labels)

print('Import completed.')

# -------------------------------------------------------------------------------------------------------------

# Classification process ---------------------------------------------------------------------------------

#knn(train_features, test_features, train_labels, test_labels)
#svm(train_features[:,:100], test_features[:,:100], train_labels, test_labels)
nnet(train_labels, test_labels, classes_list)
#nnet_feat(train_features, test_features, train_labels, test_labels, classes_list)

print('Processing time: %d min.' % int((time.time() - start_time) / 60))
