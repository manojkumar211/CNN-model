import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model=load_model("C:/AI&ML Engineer/Projects/CNN/cnnmodel.h5")

test_image=Image.open("C:/AI&ML Engineer/Projects/CNN/single/single_prediction/cat_or_dog_1.jpg")

# Data Preprocessing:-
test_image=test_image.resize((64,64))
test_image=np.array(test_image)
test_image=np.expand_dims(test_image,axis=0)

# Prediction
result=model.predict(test_image)

# Evatuation:-

if result[0][0]==1:
    print("Dog")
else:
    print("Cat")
    