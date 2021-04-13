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

