import sys
import numpy as np
import cv2
from PIL import Image
import skimage.filters
import time


# Detección de estructuras tubulares (vasos sanguíneos) usando matrices Hessianas y valores propios.
def Frangi(img):
    img_filtrada = skimage.filters.frangi(
        img, sigmas=range(1, 10, 2), scale_range=None, scale_step=None,
        alpha=0.5, beta=0.5, gamma=None, black_ridges=True, mode='reflect', cval=0
    )
    return img_filtrada


# Detección de bordes y estructuras con gradiente de Sobel.
def Evidence(img):
    img_filtrada = skimage.filters.sobel(img)
    return img_filtrada


# Filtra la imagen, resalta los píxeles significativos, y facilita la detección de bordes.
def Prunning(img):
    T = 1 / 14
    kernel = np.array([[1, 1, 1], 
                       [1, 8, 1], 
                       [1, 1, 1]])  
    kernel = np.multiply(kernel, T)
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered


# Algoritmo de segmentación principal.
def algoritmo_segmentacion():
    # Ruta de la imagen pasada como argumento.
    ruta_imagen = sys.argv[1]

    # Abrir la imagen y convertirla a escala de grises.
    imagen_editada = Image.open(ruta_imagen).convert("L")
    imagen = np.array(imagen_editada)

    t_inicio = time.time()

    # Reducir tamaños para diferentes escalas.
    ratio1, ratio2, ratio3 = 3.5, 4.5, 5.5
    w1, w2, w3 = 0.25, 0.55, 0.55

    alfa1 = cv2.resize(imagen, (int(imagen.shape[1] / ratio1), int(imagen.shape[0] / ratio1)))
    alfa2 = cv2.resize(imagen, (int(imagen.shape[1] / ratio2), int(imagen.shape[0] / ratio2)))
    alfa3 = cv2.resize(imagen, (int(imagen.shape[1] / ratio3), int(imagen.shape[0] / ratio3)))

    # Aplicar los filtros Frangi y Sobel, y combinarlos.
    f1, e1 = Frangi(alfa1), Evidence(alfa1)
    alfa1 = w1 * f1 + w3 * e1

    f2, e2 = Frangi(alfa2), Evidence(alfa2)
    alfa2 = w1 * f2 + w3 * e2

    f3, e3 = Frangi(alfa3), Evidence(alfa3)
    alfa3 = w1 * f3 + w3 * e3

    # Reescalar las imágenes filtradas a su tamaño original.
    alfa1 = cv2.resize(alfa1, (imagen.shape[1], imagen.shape[0]))
    alfa2 = cv2.resize(alfa2, (imagen.shape[1], imagen.shape[0]))
    alfa3 = cv2.resize(alfa3, (imagen.shape[1], imagen.shape[0]))

    # Combinar las escalas.
    seg = np.maximum.reduce([alfa1, alfa2, alfa3])

    # Filtrado final (Prunning).
    final = Prunning(seg)

    # Convertir la imagen final a un formato guardable.
    final = (final * 255).astype('uint8')

    # Sobrescribir la imagen de entrada.
    cv2.imwrite(ruta_imagen, final)
    print("Procesamiento completo. Duración:", time.time() - t_inicio, "s")


if __name__ == "__main__":
    algoritmo_segmentacion()
