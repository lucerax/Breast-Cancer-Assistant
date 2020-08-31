
"""
Created on Sat May 30 19:34:25 2020
"""

import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet201
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools

def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE,RESIZE))
            IMG.append(np.array(img))
    return IMG

benign_train = np.array(Dataset_loader(os.path.join('bk_data', 'train', 'benign'), 224))
malign_train = np.array(Dataset_loader(os.path.join('bk_data', 'train', 'malignant'), 224))
benign_test = np.array(Dataset_loader(os.path.join('bk_data', 'test', 'benign'), 224))
malign_test = np.array(Dataset_loader(os.path.join('bk_data', 'test', 'malignant'), 224))

benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train,
    test_size=0.2,
    random_state=11
)

"""
w=60
h=40
fig=plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if np.argmax(Y_train[i]) == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(x_train[i], interpolation='nearest')
plt.show()
"""

BATCH_SIZE = 2

train_generator = ImageDataGenerator(
        zoom_range=2,  # set range for random zoom
        rotation_range = 90,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from model import *

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
def cnn(img_shape=(224, 224, 3)):

    classifier = Sequential()
    #group A
    #classifier.add(ZeroPadding2D(padding=(1, 1)))
    classifier.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = img_shape))
    classifier.add(Conv2D(filters = 32, kernel_size = (3,3)))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    #group B
    classifier.add(ZeroPadding2D(padding=(1, 1)))
    classifier.add(Conv2D(filters = 64, kernel_size = (3,3)))
    classifier.add(Conv2D(filters = 64, kernel_size = (3,3)))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

    #FCN
    classifier.add(Flatten())
    classifier.add(Dense(activation = 'relu', units = 512))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(activation = 'relu', units = 512))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(activation = 'sigmoid', units = 2))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#model = cnn(img_shape = (32, 32, 3))
#input_shape = (None, 32, 32, 3)
#model.build(input_shape)
#model.summary()


def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    return model

resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(resnet ,lr = 1e-4)
model.summary()


checkpoint_path = "dense.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath = checkpoint_path,
    save_best_only = True,
    save_weights_only = True,
    verbose = 1)
learn_control = ReduceLROnPlateau(monitor='val_acc', patience=5,
                                  verbose=1,factor=0.2, min_lr=1e-7)



#train the model with new callback
history = model.fit_generator(
    train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[learn_control, checkpoint]
    )



model.save('my_model2.h5')

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
tfback._get_available_gpus()

from keras import backend as K
from keras import models
import tensorflow as tf
with K.tf.device('/gpu:1'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=4, allow_soft_placement=True,\
           device_count = {'CPU' : 1, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)

#CONTINUATION OF TRAINING
new_model = models.load_model('keras-flask-deploy-webapp/models/my_model2.h5')
history = new_model.fit_generator(
    train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[learn_control, checkpoint],
    workers=100,
    max_queue_size=100
    )
new_model.save('dense_model2.h5')

single_path = os.path.join('test_images')
single = np.array(Dataset_loader(single_path, 224))
temp = new_model.predict(single)
Y_pred = new_model.predict(X_test)
model = models.load_model('keras-flask-deploy-webapp/models/my_model2.h5')
Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))

cm_plot_label =['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_label, title ='Confusion Metrix for Breast Cancer')

precision = cm[1][1]/(cm[0][1] + cm[1][1])
recall = cm[1][1]/(cm[1][1] + cm[1][0])
specificity = cm[0][0]/(cm[0][0] + cm[1][0])
accuracy = (cm[0][0] + cm[1][1])/np.sum(cm)
f1 = (2* recall * precision)/(recall + precision)
print("precision = ", precision)
print("recall/sensitivity = ", recall)
print("specificity = ", specificity)
print("accuracy = ", accuracy)
print ("f1 = ", f1)
