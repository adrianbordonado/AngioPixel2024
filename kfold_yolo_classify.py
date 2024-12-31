from pathlib import Path
from ultralytics import YOLO
import yaml
from sklearn.model_selection import KFold
import os

dataset_path = Path("C:/Users/adria/Downloads/CADICA/dataset/")  # replace with 'path/to/dataset' for your custom data

from collections import Counter

ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results
    
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
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)

import shutil
from pathlib import Path
from sklearn.model_selection import KFold


# Configuración de K-Fold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Combinar imágenes de ambas clases
all_images = images_lesion + images_nonlesion
labels = ["lesion"] * len(images_lesion) + ["nonlesion"] * len(images_nonlesion)

"""
# Iterar sobre cada fold
for fold, (train_index, val_index) in enumerate(kf.split(all_images)):
    print(f"Procesando Fold {fold + 1}...")

    # Crear carpetas para este fold
    train_lesion_dir = save_path / f"fold_{fold + 1}" / "train" / "lesion"
    val_lesion_dir = save_path / f"fold_{fold + 1}" / "val" / "lesion"
    train_nonlesion_dir = save_path / f"fold_{fold + 1}" / "train" / "nonlesion"
    val_nonlesion_dir = save_path / f"fold_{fold + 1}" / "val" / "nonlesion"
    
    for directory in [train_lesion_dir, val_lesion_dir, train_nonlesion_dir, val_nonlesion_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copiar imágenes según los índices de entrenamiento y validación
    for idx in train_index:
        image = all_images[idx]
        label = labels[idx]
        if label == "lesion":
            shutil.copy(image, train_lesion_dir / image.name)
        else:
            shutil.copy(image, train_nonlesion_dir / image.name)

    for idx in val_index:
        image = all_images[idx]
        label = labels[idx]
        if label == "lesion":
            shutil.copy(image, val_lesion_dir / image.name)
        else:
            shutil.copy(image, val_nonlesion_dir / image.name)
    
    print(f"Fold {fold + 1} completado.")
   """  

results = {}
matrices=[]

model = YOLO("runs/classify/train2/weights/best.pt")
# Define your additional arguments here
batch = 16
project = "kfold_demo"

for k in range(ksplit):

    model.val(source=str(save_path)+"/fold_"+str(k), save=False, verbose=False)  # includes any train arguments
    results[k] = model.metrics  # save output metrics for further analysis
    matrices.append(model.metrics.confusion_matrix.matrix)
    
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calcular_metricas(confusion_matrices):
    # Inicializamos listas para las métricas
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    
    # Iteramos sobre cada matriz de confusión
    for cm in confusion_matrices:
        # Extraemos TP, TN, FP, FN de la matriz de confusión
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        # Calculamos Precision, Recall, F1 Score y AUC
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        # Almacenamos los resultados
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Devolvemos los resultados como listas
    return {
        'Precision': precision_list,
        'Recall': recall_list,
        'F1 Score': f1_list
    }


# Calculamos las métricas
resultados = calcular_metricas(matrices)

# Imprimimos los resultados
for metric, values in resultados.items():
    print(f"{metric}: {values}")




