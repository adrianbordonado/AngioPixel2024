import os
import shutil
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from ultralytics import YOLO

# Función para dividir datos y etiquetas en pliegues
def split_data_for_cross_validation(image_paths, labels, n_splits=5, stratified=False):
    if stratified:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = skf.split(image_paths, labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = kf.split(image_paths)
    return list(splits)

# Simulación del entrenamiento y validación del modelo
def train_and_validate_yolo(train_images, train_labels, val_images, val_labels, fold_num):
    print("Validando para el pliegue",fold_num)
    modelo = YOLO("runs/detect/train4/weights/best.pt")
    result= modelo.val(data="dataset_detect/config.yaml", save=False, verbose= False).results_dict

    return {"mAP": result.get("metrics/mAP50(B)"), "Precision": result.get("metrics/precision(B)"), "Recall": result.get("metrics/recall(B)")}

image_paths=[]
directorio= "dataset_detect/images/val/"
# Preparar rutas de imágenes y etiquetas (ejemplo)
for image_path in os.listdir(directorio):
    image_paths.append(directorio+image_path)
labels = np.random.randint(0, 2, len(image_paths))  # Etiquetas binarias para StratifiedKFold

# Realizar K-Fold y Stratified K-Fold Cross Validation
n_splits = 5
splits = split_data_for_cross_validation(image_paths, labels, n_splits=n_splits, stratified=True)

# Resultados finales
results = []

for fold, (train_idx, val_idx) in enumerate(splits):
    train_images = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    # Entrenar y validar para cada pliegue
    metrics = train_and_validate_yolo(train_images, train_labels, val_images, val_labels, fold + 1)
    results.append(metrics)

# Mostrar resultados
for fold_num, metrics in enumerate(results, start=1):
    print("Resultados para el pliegue", fold_num,":", metrics)
