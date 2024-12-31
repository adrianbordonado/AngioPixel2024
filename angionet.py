import sys
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
from deeplab_model import BilinearUpsampling  # Importar la capa personalizada

# Ruta al modelo AngioNet
MODEL_PATH = "angionet.keras"

def preprocess_image(image_path):
    """Carga y preprocesa la imagen para el modelo AngioNet."""
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {image_path}")
    
    # Redimensionar la imagen a 512x512x1 (tamaño esperado por AngioNet)
    image_resized = cv2.resize(image, (512, 512))  # Sin especificar el número de canales
    
    # Normalizar la imagen (escala de valores entre 0 y 1)
    image_normalized = image_resized / 255.0
    
    # Expande las dimensiones para que sea compatible con el modelo
    # Forma: (1, 512, 512, 1) (1 imagen en batch, 512x512, 1 canal)
    image_expanded = np.expand_dims(image_normalized, axis=-1)  # (512, 512, 1)
    image_expanded = np.expand_dims(image_expanded, axis=0)  # (1, 512, 512, 1)
    
    return image_expanded

def predict(image_path):
    """Utiliza el modelo AngioNet para predecir si hay lesión."""
    # Cargar el modelo AngioNet con la capa personalizada
    model = load_model(MODEL_PATH, custom_objects={'BilinearUpsampling': BilinearUpsampling})

    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_path)

    # Realizar la predicción
    predictions = model.predict(preprocessed_image)

    # Obtener la clase predicha y el porcentaje de confianza
    # Para cada píxel, obtener la clase con mayor probabilidad
    predicted_class = np.argmax(predictions, axis=-1)  # (1, 512, 512) con el índice de la clase más probable

    # Obtener la confianza de la clase predicha para cada píxel
    confidence = np.max(predictions, axis=-1) * 100  # (1, 512, 512) con el porcentaje de confianza

    # Calcular el porcentaje de confianza promedio de la imagen
    average_confidence = np.mean(confidence)

    # Interpretar la predicción para la clase más probable en general
    # Si la mayoría de los píxeles tienen la clase 0 (sin lesión), consideramos que es "No tiene lesión"
    if np.mean(predicted_class) == 0:
        result = "No tiene lesión"
    else:
        result = "Tiene lesión"

    return result, average_confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <ruta_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        result, confidence = predict(image_path)
        print("Modelo: angionet")
        print(f"Prediccion: {result}")
        print(f"Confianza promedio: {confidence:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

