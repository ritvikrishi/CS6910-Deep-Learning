# importing required packages
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization, SpatialDropout2D
import matplotlib.pyplot as plt
import os
import PIL
import glob
import zipfile
import pathlib

# # data import for running on colab
# dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
# data_dir = tf.keras.utils.get_file('/root/inature_12k.zip', origin=dataset_url, extract=False)

# with zipfile.ZipFile(data_dir, 'r') as zip_ref:
#     zip_ref.extractall('/content')

data_dir = '/content/inaturalist_12K'  # change path accordingly
data_all =  data_dir + '/train'
data_test = data_dir + '/val'

# checking data
data_path = pathlib.Path(data_dir)
image_count = len(list(data_path.glob('//*.jpg')))
print(image_count)  # should print 11999

batch_size = 32
img_height = 150 
img_width = 150  
num_classes = 10

# getting image dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, validation_split=0.1,
                            subset="training", seed=123,
                            image_size=(img_height, img_width), batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(trin_dir, validation_split=0.1,
                            subset="validation", seed=123,
                            image_size=(img_height, img_width), batch_size=batch_size)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, seed=123,
                            image_size=(img_height, img_width), batch_size=batch_size)


# pre-fetching data for faster access
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# defining data augmentation
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomTranslation((-0.2,0.2), (-0.2, 0.2))
  ]
)

# building model
def buildCNN(nFilters, ksize=[3,3,3,3,3], filterFac=1, isDataAug=False, dropout=0.0, isBN=True, denseN=256):
    model = Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
    if isDataAug:
        model.add(data_augmentation)
    for i in range(5):
        model.add(Conv2D(math.ceil((filterFac**i) * nFilters[i]), (ksize[i], ksize[i]), padding = 'same'))
        if isBN:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(dropout))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(denseN))
    model.add(Activation('relu'))
    model.add(Dense(10, activation = 'softmax'))
    return model

# manual training
nFilters = [64, 64, 64, 32, 32]
ksize=[3,3,3,3,3]
epochs=10
tf.keras.backend.clear_session()
model = buildCNN(nFilters = nFilters, ksize=ksize, dropout = 0.3, isBN=True, isDataAug=False, denseN=128)
model.compile(optimizer = 'adamax', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

history = model.fit(train_ds,  validation_data=val_ds, epochs=epochs)

# # plotting 
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()