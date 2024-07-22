import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from maxpool import classifier
from keras.layers import MaxPooling2D
from keras.layers import Flatten



classifier.add(Flatten())