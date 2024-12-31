import sys
from PIL import Image
import cv2
import numpy as np

def LoG():
    # Ruta de la imagen pasada como argumento.
    ruta_imagen = sys.argv[1]

    # Abrir la imagen
    imagen_editada = Image.open(ruta_imagen)
    img = np.array(imagen_editada)

    # Aplicar GaussianBlur y Laplacian
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # Normalizar el resultado para que esté en el rango 0-255
    log_normalized = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX)
    log_normalized = log_normalized.astype(np.uint8)

    # Guardar la imagen procesada
    cv2.imwrite(ruta_imagen, log_normalized)

    return

if __name__ == "__main__":
    LoG()
