import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from model import classifier
from data import train_set,test_set
from keras.layers import Dense

test_image=Image.open("C:/AI&ML Engineer/Projects/CNN/single/single_prediction/cat_or_dog_1.jpg")

# Data Preprocessing:-
test_image=test_image.resize((64,64))
test_image=np.array(test_image)
test_image=np.expand_dims(test_image,axis=0)

# Prediction
model="cnnmodel.h5"
result=model.predict(test_image)

# Evatuation:-

if result[0][0]==1:
    print("Dog")
else:
    print("Cat")