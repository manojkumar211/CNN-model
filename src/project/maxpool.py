import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from cnn import classifier
from keras.layers import MaxPooling2D


classifier.add(MaxPooling2D(pool_size=2,strides=2))