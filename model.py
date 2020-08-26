# -*- coding: utf-8 -*-
"""
Builds the CNN model. Currently, 2 conv/conv/pool blocks are used 
followed by 3 layer FCN
"""

# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
def cnn(img_shape=(224, 224, 3)):
    
    classifier = Sequential()
    #group A
    classifier.add(ZeroPadding2D(padding=(1, 1)))
    classifier.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = img_shape))
    classifier.add(Conv2D(filters = 32, kernel_size = (3,3)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    #group B
    classifier.add(ZeroPadding2D(padding=(1, 1)))
    classifier.add(Conv2D(filters = 64, kernel_size = (3,3)))
    classifier.add(Conv2D(filters = 64, kernel_size = (3,3)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    
    #FCN
    classifier.add(Flatten())
    classifier.add(Dense(activation = 'relu', units = 512))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(activation = 'relu', units = 512))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(activation = 'sigmoid', units = 1))

    return classifier

model = cnn(img_shape = (32, 32, 3))
input_shape = (None, 32, 32, 3)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.build(input_shape)
model.summary()




