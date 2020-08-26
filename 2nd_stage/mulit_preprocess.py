#SET UP DIRECTORY 
import os
import numpy as np
import shutil
import random

# # Creating Train / Val / Test folders (One time use)
def split_dir(new_root, cur_dir, classes, test_ratio, val_ratio=0):
  for c in classes:
    os.makedirs(os.path.join(new_root, 'train', c))
    if val_ratio:
      os.makedirs(os.path.join(new_root, 'val', c))
    os.makedirs(os.path.join(new_root, 'test', c))
    list_files = []
    src = os.path.join(cur_dir, c)# Folder to copy images from
    for root, dirs, files in os.walk(src):
        for f in files:
            path = os.path.join(root, f)
            print(path)
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
        shutil.copy(name, os.path.join(new_root, 'train', c))

    if val_ratio:
      for name in val_FileNames:
          shutil.copy(name, os.path.join(new_root, 'val', c))

    for name in test_FileNames:
        shutil.copy(name, os.path.join(new_root, 'test', c))


split_dir('binary_diff', 'webpath', ['benign', 'malignant'], test_ratio=0.2)
split_dir('malignant_diff', os.path.join('webpath', 'malignant'), ['adenoid_cystic_carc', 'apocrine_carc', 'DCIS', 'IDC', 'ILC', 'LCIS', 'lymphoma', 'medullary_carc', 'metaplastic_carc', 'mucinous_carc', 'paget_carc', 'papillary_carc', 'sarcoma', 'secretory_carc', 'tubular_carc'], test_ratio=0.2)
split_dir('benign_diff', os.path.join('webpath', 'benign'), ['fibroadenoma', 'papilloma', 'phyllodes', 'reactive', 'tubular_adenoma'], test_ratio=0.2)
