import pandas as pd 
import os 
import tensorflow as tf
import keras
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Declaracion de vectores de trabajo
Categories=['lesion','nonlesion'] 
img_arr=[] #input array 
target_arr=[] #output array 
datadir='dataset/train/' 
#path which contains all the categories of images 

#Insercion en el vector de imagenes con imagenes redimensionadas y etiquetadas
for i in Categories: 
      
    print(f'loading... category : {i}') 
    path=os.path.join(datadir,i) 
    for img in os.listdir(path): 
        imagen=imread(os.path.join(path,img)) 
        img_resized=resize(imagen,(224,224,3)) 
        target_arr.append(Categories.index(i))
        img_arr.append(img_resized) 
    print(f'loaded category:{i} successfully') 

print("Ha llegado hasta aquí")
X=np.array(img_arr, dtype='float32') 
y=np.array(target_arr, dtype='int32')

# Normalize the image data (scaling pixel values to [0, 1])
X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Split hecho")


# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary classification
              metrics=['accuracy'])

print("Modelo compilado")

# Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=18,
                    validation_data=(X_val, y_val))



model.save("cnn.keras")

# Make predictions on the validation set
y_pred = model.predict(X_val)
y_pred_class = (y_pred > 0.5).astype("int32")  # Convert the predictions to binary (0 or 1)

# Evaluate the predictions (optional)
from sklearn.metrics import classification_report, confusion_matrix

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_val, y_pred_class))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_class))
