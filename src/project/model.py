import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from flat import classifier
from data import train_set,test_set
from keras.layers import Dense

# Input Layer
classifier.add(Dense(units=128,activation="relu"))

# Output Layer
classifier.add(Dense(units=1,activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(x=train_set,validation_data=test_set,epochs=25)