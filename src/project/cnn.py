import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Conv2D


classifier=tf.keras.models.Sequential()

classifier.add(Conv2D(input_shape=[64,64,3],filters=32,kernel_size=3,activation="relu"))

