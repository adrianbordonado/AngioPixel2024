import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from skimage.io import imread 
from skimage.transform import resize
import os

dataset_path = Path("C:/Users/adria/Downloads/CADICA/dataset/")  # replace with 'path/to/dataset' for your custom data


ksplit = 5
    
import datetime

supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images_lesion = []
images_nonlesion=[]

import pandas as pd
# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images_lesion.extend(sorted((dataset_path / "lesion").rglob(f"*{ext}")))
    images_nonlesion.extend(sorted((dataset_path / "nonlesion").rglob(f"*{ext}")))
    
# Create the necessary directories and dataset YAML files (unchanged)
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val_SVM")
save_path.mkdir(parents=True, exist_ok=True)

import shutil
from pathlib import Path

kf = KFold(n_splits=ksplit, shuffle=True, random_state=42)

# Combinar imágenes de ambas clases
all_images = images_lesion + images_nonlesion
labels = ["lesion"] * len(images_lesion) + ["nonlesion"] * len(images_nonlesion)


# Iterar sobre cada fold
for fold, (train_index, val_index) in enumerate(kf.split(all_images)):
    print(f"Procesando Fold {fold + 1}...")

    # Crear carpetas para este fold
    lesion_dir = save_path / f"fold_{fold + 1}" / "lesion"
    nonlesion_dir = save_path / f"fold_{fold + 1}" / "nonlesion"
    
    for directory in [lesion_dir, nonlesion_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copiar imágenes según los índices de entrenamiento y validación
    for idx in train_index:
        image = all_images[idx]
        label = labels[idx]
        if label == "lesion":
            shutil.copy(image, lesion_dir / image.name)
        else:
            shutil.copy(image, nonlesion_dir / image.name)
            
    for idx in val_index:
        image = all_images[idx]
        label = labels[idx]
        if label == "lesion":
            shutil.copy(image, lesion_dir / image.name)
        else:
            shutil.copy(image, nonlesion_dir / image.name)
    
    print(f"Fold {fold + 1} completado.")
     


def get_x_y(datadir):
    
    flat_data_arr=[]
    target_arr=[]
    #Insercion en el vector de imagenes con imagenes redimensionadas y etiquetadas
    for i in ["lesion","nonlesion"]: 
          
        print(f'loading... category : {i}') 
        path=os.path.join(datadir,i) 
        for img in os.listdir(path): 
            img_array=imread(os.path.join(path,img)) 
            img_resized=resize(img_array,(150,150,3)) 
            flat_data_arr.append(img_resized.flatten()) 
            target_arr.append(["lesion","nonlesion"].index(i)) 
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
    
    return x, y

model= joblib.load("svm.pkl")

batch = 16
project = "kfold_demo"
epochs = 1
split_folders = sorted(os.listdir(save_path))

for i, test_folder in enumerate(sorted(os.listdir(save_path))):

    print(f"Iteración {i + 1}: Usando {test_folder} como carpeta de prueba")

    # Identificar las carpetas de entrenamiento y prueba
    test_folder_path = os.path.join(save_path, test_folder)
    train_folders = [os.path.join(save_path, f) for f in split_folders if f != test_folder]

    # Cargar datos de prueba
    x_test, y_test = get_x_y(test_folder_path)

    # Cargar datos de entrenamiento de todas las demás carpetas
    x_train = []
    y_train = []
    
    for train_folder in train_folders:
        images, labels =get_x_y(train_folder)
        x_train.append(images)
        y_train.append(labels)
    
    # Combinar las carpetas de entrenamiento
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Mezclar los datos de entrenamiento
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Realizar predicciones con el modelo cargado
    y_pred = model.predict(x_test.reshape(len(x_test), -1))  # Asegúrate de aplanar si es necesario
    
    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {accuracy * 100:.2f}%")

    # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred, target_names=['lesion', 'nonlesion']))


        
