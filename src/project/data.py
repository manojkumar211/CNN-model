import numpy as np
import pandas as pd
import tensorflow as tf
import keras


train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2)

train_set=train_datagen.flow_from_directory("C:/AI&ML Engineer/Projects/CNN/training/training_set",target_size=(64,64),class_mode="binary")

print(train_set.class_indices)


test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2)

test_set=test_datagen.flow_from_directory("C:/AI&ML Engineer/Projects/CNN/test/test_set",target_size=(64,64),class_mode="binary")

test_set.class_indices

