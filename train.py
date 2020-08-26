"""
Created on Sat May 30 13:37:31 2020

@author: aniru
"""

from model import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

checkpoint_path = "train_ckpt/cp.ckpt"
cp_callback = ModelCheckpoint(
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



model.save('my_model.h5')
