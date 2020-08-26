# -*- coding: utf-8 -*-
"""
This script creates the training, testing and validation directories and populates
them with benign and malignant data
"""
from model import *
import keras

import os
import numpy as np
import shutil
import random

# # Creating Train / Val / Test folders (One time use)
root_dir = 'bk_data'
diagnosis_dir = ['benign', 'malignant']
classes = {
    'benign':['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma'],
    'malignant': ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']}

val_ratio = 0.2
test_ratio = 0.1
#TODO: even out classes assigned to training and testing
for diag in diagnosis_dir:
    os.makedirs(os.path.join(root_dir, 'train', diag))
    os.makedirs(os.path.join(root_dir, 'val', diag))
    os.makedirs(os.path.join(root_dir, 'test', diag))
    top = os.getcwd()
    print(top)
    list_files = []
    for cls in classes[diag]:
    # Creating partitions of the data after shuffeling
        print(cls)
        src = os.path.join(top, 'breast', diag, 'SOB', cls)# Folder to copy images from
        for root, dirs, files in os.walk(src):
            for f in files:
                path = os.path.join(root, f)
                list_files.append(path)


    np.random.shuffle(list_files)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(list_files),
                                                              [int(len(list_files)* (1 - val_ratio + test_ratio)),
                                                               int(len(list_files)* (1 - test_ratio))])



    print('Total images: ', len(list_files))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, os.path.join(root_dir, 'train', diag))

    for name in val_FileNames:
        shutil.copy(name, os.path.join(root_dir, 'val', diag))

    for name in test_FileNames:
        shutil.copy(name, os.path.join(root_dir, 'test', diag))
