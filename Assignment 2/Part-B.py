# importing required packages
import numpy as np
import pandas as pd
import tensorflow as tf

import math
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import os
import PIL
import glob
import zipfile
import pathlib
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50 
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_InceptionV3
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_Xception
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_Vgg19
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_Efficientnet_b3
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_Efficientnet_b7

# # data import for colab
# dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
# data_dir = tf.keras.utils.get_file('/root/inature_12k.zip', origin=dataset_url, extract=False)

# with zipfile.ZipFile(data_dir, 'r') as zip_ref:
#     zip_ref.extractall('/content')

# change path accordingly
data_dir = '/content/inaturalist_12K'
data_all =  data_dir + '/train'
data_test = data_dir + '/val'

#checking import
data_path = pathlib.Path(data_dir)
image_count = len(list(data_path.glob('*/*/*.jpg')))
print(image_count)

BATCH_SIZE = 32

model_pixel_map = {
    "ResNet50":224,
    "InceptionResNetV2":299,
    "InceptionV3":299,
    "Xception":299,
    "Vgg19":224,
    "Efficientnet_b3": 300,
    "Efficientnet_b7": 600,
}

def get_img_size(model_name):
    return (model_pixel_map[model_name],model_pixel_map[model_name], 3)

datagen_kwargs={
    "ResNet50": dict(validation_split=.10, preprocessing_function=preprocess_input_ResNet50),
    "InceptionResNetV2":dict(validation_split=.10, preprocessing_function=preprocess_input_InceptionResNetV2),
    "InceptionV3": dict(validation_split=.10, preprocessing_function=preprocess_input_InceptionV3),
    "Xception": dict(validation_split=.10, preprocessing_function=preprocess_input_Xception),
    "Vgg19": dict(validation_split=.10, preprocessing_function=preprocess_input_Vgg19),
    "Efficientnet_b3": dict(validation_split=.10, preprocessing_function=preprocess_input_Efficientnet_b3),
    "Efficientnet_b7": dict(validation_split=.10, preprocessing_function=preprocess_input_Efficientnet_b7),
}

model_dict={
    'ResNet50': tf.keras.applications.ResNet50, #weights="imagenet",input_shape=get_img_size(ResNet50),include_top=False,
    'Xception':keras.applications.Xception, #(weights="imagenet",input_shape=get_img_size(Xception),include_top=False,),
    'InceptionV3':keras.applications.InceptionV3, #(weights="imagenet",input_shape=get_img_size(InceptionV3),include_top=False,),
    'InceptionResNetV2':keras.applications.InceptionResNetV2, #(weights="imagenet",input_shape=get_img_size(InceptionResNetV2),include_top=False,),
    'Vgg19':keras.applications.VGG19, #(weights="imagenet",input_shape=get_img_size(Vgg19),include_top=False,),
    'Efficientnet_b3':tf.keras.applications.EfficientNetB3,
    'Efficientnet_b7':tf.keras.applications.EfficientNetB7,
}

def get_base_model(model_name):
  return model_dict[model_name](weights="imagenet",input_shape=get_img_size(model_name),include_top=False,)

def preproc(model_name):
    img_size=(model_pixel_map[model_name],model_pixel_map[model_name])
    datagen_kwarg=datagen_kwargs[model_name]
    dataflow_kwarg=dict(target_size=img_size,batch_size=BATCH_SIZE)
    return datagen_kwarg,dataflow_kwarg
