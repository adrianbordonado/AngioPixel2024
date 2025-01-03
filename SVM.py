#Librerias de trabajo
import pandas as pd 
import os 
import tensorflow
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

#Declaracion de vectores de trabajo
Categories=['lesion','nonlesion'] 
flat_data_arr=[] #input array 
target_arr=[] #output array 
datadir='dataset/train/' 
#path which contains all the categories of images 

#Insercion en el vector de imagenes con imagenes redimensionadas y etiquetadas
for i in Categories: 
      
    print(f'loading... category : {i}') 
    path=os.path.join(datadir,i) 
    for img in os.listdir(path): 
        img_array=imread(os.path.join(path,img)) 
        img_resized=resize(img_array,(150,150,3)) 
        flat_data_arr.append(img_resized.flatten()) 
        target_arr.append(Categories.index(i)) 
    print(f'loaded category:{i} successfully') 
flat_data=np.array(flat_data_arr) 
target=np.array(target_arr)

#Creación del dataFrame de imagenes
df=pd.DataFrame(flat_data)  
df['Target']=target 
df.shape

#input data  
x=df.iloc[:,:-1]  
#output data 
y=df.iloc[:,-1]
print("Split por hacer")
# Splitting the data into training and testing sets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, 
                                               random_state=77, 
                                               stratify=y) 

print("Split hecho")
# Defining the parameters grid for GridSearchCV 
param_grid={'C':[0.1,1], 
            'gamma':[0.1,1], 
            'kernel':['rbf','poly']} 
  
print("Crear mordelo")
# Creating a support vector classifier 
svc=svm.SVC(probability=True) 

param = {'kernel' : ['linear'],'C' : [5],'degree' : [3],'coef0' : [0.5],'gamma' : ['auto']}
  
# Creating a model using GridSearchCV with the parameters grid 
model=GridSearchCV(svc,param)
print("Entrenar modelo")
# Training the model using the training data 
model.fit(x_train,y_train)

print("Testear modelo")
# Testing the model using the testing data 
y_pred = model.predict(x_test) 
  
# Calculating the accuracy of the model 
accuracy = accuracy_score(y_pred, y_test) 

import joblib

#save your model or results
joblib.dump(model, 'model_file_name.pkl')

#load your model for further usage
#joblib.load("model_file_name.pkl")
# Print the accuracy of the model 
print(f"The model is {accuracy*100}% accurate")

print(classification_report(y_test, y_pred, target_names=['lesion', 'nonlesion']))

