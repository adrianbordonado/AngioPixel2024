import sys
from PIL import Image
import skimage.filters
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

def Gradient():
    # Ruta de la imagen pasada como argumento.
    ruta_imagen = sys.argv[1]

    # Abrir la imagen
    imagen_editada = Image.open(ruta_imagen)
    imagen = np.array(imagen_editada)

    # Aplicar gradiente
    gradient = skimage.filters.sobel(imagen)

    # Normalizar el gradiente a un rango de 0-255 para guardarlo como imagen.
    gradient_normalized = (gradient * 255 / gradient.max()).astype(np.uint8)

    # Sobrescribir la imagen de entrada.
    cv2.imwrite(ruta_imagen, gradient_normalized)
    
    return

if __name__ == "__main__":
    Gradient()
