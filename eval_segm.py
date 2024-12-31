from segmentado_multiescala import algoritmo_segmentacion
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageTk, ImageFilter, ImageOps
from ultralytics import YOLO
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection/union if union > 0 else 0

def evaluate_segmentation_metrics(auto_mask, manual_mask):
    # Verificar dimensiones
    if auto_mask.shape != manual_mask.shape:
        raise ValueError("Las máscaras deben tener las mismas dimensiones.")

    # Asegurarse de que las máscaras sean binarias
    auto_mask = (auto_mask > 0).astype(np.uint8)
    manual_mask = (manual_mask > 0).astype(np.uint8)

    # Aplanar las máscaras para métricas basadas en sklearn
    auto_mask_flat = auto_mask.flatten()
    manual_mask_flat = manual_mask.flatten()

    # Cálculo de métricas
    iou = calculate_iou(auto_mask, manual_mask)
    precision = precision_score(manual_mask_flat, auto_mask_flat)
    recall = recall_score(manual_mask_flat, auto_mask_flat)
    f1 = f1_score(manual_mask_flat, auto_mask_flat)
    accuracy = accuracy_score(manual_mask_flat, auto_mask_flat)

    # Devolver resultados
    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy
    }



#GENERAR MÁSCARAS A PARTIR DE DATASET
for archivo in os.listdir("../DIAS/training/images/"):
    
    image = Image.open("../DIAS/training/images/"+archivo)
    image = image.resize((512, 512))
    image = algoritmo_segmentacion(np.array(image))
    cv2.imwrite('../DIAS/mascaras_algoritmo/'+archivo, image)
    


#COMPARAR MÁSCARAS CON LAS DEL DATASET DIAS

mascaras=[]
    
#CADA ELEMENTO MÁSCARAS CONTIENE UNA PAREJA DE MÁSCARAS, LA PRIMERA SIENDO LA GENERADA AUTOMÁTICAMENTE Y LA SEGUNDA LA MANUALMENTE

resultados=[]

#RECORTAMOS TODAS LAS MÁSCARAS PARA QUITAR BORDES

for archivo in os.listdir("../DIAS/mascaras_algoritmo/"):
    # Abrir la imagen usando PIL
    image = Image.open("../DIAS/mascaras_algoritmo/" + archivo)

    # Convertir la imagen a un array de numpy y recortar los bordes
    image = np.array(image)[30:-30, 30:-30]

    # Convertir el array recortado de vuelta a imagen (PIL)
    image = Image.fromarray(image)

    # Redimensionar la imagen a 512x512
    image = image.resize((512, 512))

    # Convertir la imagen redimensionada a un array de numpy (para que OpenCV pueda manejarla)
    image = np.array(image)

    # Asegurarse de que la imagen tenga tipo de dato uint8 antes de guardarla
    image = np.uint8(image)

    # Guardar la imagen utilizando OpenCV
    cv2.imwrite('../DIAS/mascaras_algoritmo/' + archivo, image)

    print(f"Imagen {archivo} procesada y guardada.")
    
for archivo in os.listdir("../DIAS/mascaras_algoritmo/"):
    
    label = "image_"+archivo.split("_")[1]+"_i0.png"
    mascaras.append([cv2.imread("../DIAS/mascaras_algoritmo/"+archivo),cv2.resize(cv2.imread("../DIAS/training/labels/"+label), (512, 512), interpolation=cv2.INTER_NEAREST)])

#CALCULAR RESULTADOS

for auto, manual in mascaras:
    
    resultados.append(evaluate_segmentation_metrics(auto, manual))
    
  
IoU=[]; precision=[]; recall=[]; f1score=[]; accuracy=[]
    
for resultado in resultados:
    
    IoU.append(resultado.get("IoU"))
    precision.append(resultado.get("Precision"))
    recall.append(resultado.get("Recall"))
    f1score.append(resultado.get("F1 Score"))
    accuracy.append(resultado.get("Accuracy"))
    