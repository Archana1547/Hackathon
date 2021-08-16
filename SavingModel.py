import pickle
import tensorflow as tf
#import matplotlib.pyplot as plt
import os
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#import seaborn as sns
import pathlib
import PIL
import numpy as np
import dlib

from keras.models import load_model

batchSize = 30
imgHeight = 180
imgWidth = 180



Class_names=['Dusky','Fair']
dataset = pathlib.Path('StrDataSet')
trainDS=tf.keras.preprocessing.image_dataset_from_directory(dataset,labels='inferred',class_names=Class_names,label_mode="int",seed=60,validation_split=0.2,subset='training',image_size=(imgHeight, imgWidth),batch_size=batchSize)
validationDS=tf.keras.preprocessing.image_dataset_from_directory(dataset,labels='inferred',class_names=Class_names,label_mode="int",seed=60,validation_split=0.2,subset='validation',image_size=(imgHeight, imgWidth),batch_size=batchSize)

num_classes = 2

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),

  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




model.summary()

epochs=10
history = model.fit(trainDS,validation_data=validationDS,epochs = epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


model.save('model.h5')

