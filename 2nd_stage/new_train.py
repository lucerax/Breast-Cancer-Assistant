# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:15:08 2020

@author: aniru
"""
import keras
from keras.preprocessing.image import ImageDataGenerator
import os



#LOAD AND PREPROCESS DATA
#With image generator

training_path = os.path.join('binary_diff', 'train')
validation_path = os.path.join('binary_diff', 'test')
batch_size = 2

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        height_shift_range=0.5,
        rotation_range = 90)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        training_path,  # this is the target directory
        target_size=(224, 224),  # all images will be resized 
        batch_size=batch_size, class_mode='categorical'
        )  

validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(224, 224),
        batch_size=batch_size,class_mode='categorical'
        )


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.layers import Input
"""
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels

model = load_model(os.path.join('keras-flask-deploy-webapp', 'models', 'my_model2.h5'))
model.summary()
new_input = Input(shape=(512, 512, 3))
delete_layer(model, model.layers[0])
insert_layer(model, model.layers[0], new_input)
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam
from keras.applications import DenseNet201
from keras.metrics import Precision, Recall
def build_model(backbone):
    model = Sequential()
    model.add(backbone)
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    return model

densenet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(densenet)
model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
model.summary()

#TODO: re-run
model.fit_generator(
        train_generator,
        steps_per_epoch=500 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=500 // batch_size,
        workers=100,
        max_queue_size=100,
        class_weight={0:2.8, 1:1.0}
        )
model.save('binary_diff.h5')
y_train = train_generator.classes





"""
malignant_diff training
"""

training_path = os.path.join('malignant_diff', 'train')
validation_path = os.path.join('malignant_diff', 'test')
batch_size = 2

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        height_shift_range=0.5,
        rotation_range = 90)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        training_path,  # this is the target directory
        target_size=(224, 224),  # all images will be resized 
        batch_size=batch_size, class_mode='categorical'
        )  

validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(224, 224),
        batch_size=batch_size,class_mode='categorical'
        )

from utils import change_model
mal_model = load_model(os.path.join('keras-flask-deploy-webapp', 'models', 'my_model2.h5'))


"""
CHANGE OUTPUT SHAPE OF LOADED MODEL
"""
from keras.models import Model
predictions = Dense(15, activation='softmax')(mal_model.layers[-2].output)
mal_model = Model(inputs=mal_model.input, outputs=predictions)
mal_model.compile(optimizer='adam', loss='categorical_crossentropy')
mal_model.summary()

"""
Account for data imbalance
"""
nums = []
for root, dirs, files in os.walk(training_path):
    count = 0
    for f in files:
        count += 1
    if count:
        nums.append(count)
mx = max(nums)
nums = [1/n*mx for n in nums]  
class_weights = {}
for i, wt in enumerate(nums):
    class_weights[i]=wt
    
"""
FIX INPUT SHAPE
"""
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
new_input = Input(shape=(512, 512, 3))

new_output = mal_model(new_input)
mal_model = Model(new_input, new_output)
mal_model.summary()mal_model = delete_layer(mal_model, mal_model.layers[0])


"""
TRAIN
"""

mal_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

mal_model.fit_generator(
        train_generator,
        steps_per_epoch=500 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=500 // batch_size,
        workers=100,
        max_queue_size=100,
        class_weight=class_weights
        )
mal_model.save('malignant_diff.h5')
