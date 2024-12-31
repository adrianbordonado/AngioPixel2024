import tkinter as tk
import win32com.client
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageEnhance
from io import BytesIO
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import csv
import cv2
import math
import matplotlib.pyplot as plt
import skimage
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import joblib
import zipfile
import time
from threading import Thread

def seleccionar_archivo():
    # Abrir el cuadro de diálogo para seleccionar imágenes
    archivos = filedialog.askopenfilenames(title="Seleccionar imágenes", filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    
    if archivos:
        # Cargar las imágenes seleccionadas
        global imagenes_originales
        global imagenes_editadas
        global resultados_globales
        global filenames
        imagenes_originales = [Image.open(archivo) for archivo in archivos]
        imagenes_editadas = imagenes_originales.copy()  # Copia de la primera imagen para aplicar filtros
        
        filenames=[]
        for img in imagenes_originales: filenames.append(img.filename.split("\\")[-1])
        
        # Inicializar el índice de la imagen actual
        global imagen_actual
        global indimagen
        
        indimagen=0
        imagen_actual = imagenes_originales[indimagen].copy()
        # Mostrar la primera imagen en la etiqueta
        mostrar_imagen(imagenes_originales[indimagen])
        
        # Habilitar los botones de exportar, deshacer y navegación
        btn_reiniciar.config(state=tk.NORMAL)
        btn_anterior.config(state=tk.NORMAL)
        btn_siguiente.config(state=tk.NORMAL)
        
        habilitar_comandos(menu_archivos)
        habilitar_comandos(menu_segmentacion)
        habilitar_comandos(menu_clasificacion)
        habilitar_comandos(menu_deteccion)
        ventana.geometry("")
        

def habilitar_comandos(menu):
    for i in range(menu.index(tk.END) + 1):
        # Obtener el comando de la posición i
        item = menu.entrycget(i, "label")  # Conseguir el texto del comando
        if item:
            # Habilitar el comando por su índice
            menu.entryconfig(i, state=tk.NORMAL)
            
def narra(texto):
    
    speaker.Speak(texto)
    
    return

def narrar_en_hilo(texto):
    Thread(target=lambda: narra(texto), daemon=True).start()
    
def mostrar_imagen(imagen):
    # Redimensionar la imagen para ajustarse a la ventana
    imagen = imagen.resize((400, 400))      
    # Convertir la imagen a un formato que Tkinter pueda mostrar
    imagen_tk = ImageTk.PhotoImage(imagen)
    
    # Mostrar la imagen en la etiqueta
    etiqueta_imagen.config(image=imagen_tk)
    etiqueta_imagen.image = imagen_tk  # Mantener una referencia a la imagen para evitar que se elimine

def Frangi(img):

    H=skimage.feature.hessian_matrix(img, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=False)
    eigen = skimage.feature.hessian_matrix_eigvals(H)
    lambdas1=[]
    lambdas2=[]
    Cv=[]
    beta=0.5
    c=0.3
    
    img_filtrada=skimage.filters.frangi(img, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=None, black_ridges=True, mode='reflect', cval=0)

    for valor in eigen[0]: lambdas2.append(valor)
    for valor in eigen[1]: lambdas1.append(valor)
    
    for i in range(len(lambdas1)):
        
        row=[]
        
        for j in range(len(lambdas1)):
        
            if(lambdas2[i][j]<lambdas1[i][j]):
                
                return False
                
            elif(lambdas2[i][j]<=0):
                
                Rb= lambdas1[i][j]/lambdas2[i][j]
                T = math.sqrt((lambdas2[i][j])**2+(lambdas1[i][j])**2)
                
                row.append(1-(math.exp(-(Rb**2)/(2*(beta**2)))*(1-math.exp(-(T**2)/(2*(c**2))))))
                
            elif(lambdas2[i][j]>0):
                
                row.append(1)
                
        Cv.append(row)
        
            
    return Cv, img_filtrada


def Koller(q,p,v):
    

    E1=v[q[0]][q[1]][np.argmin(abs(v[q[0]][q[1]]))]
    E2=v[p[0]][p[1]][np.argmin(abs(v[p[0]][p[1]]))]
    
    try:
            
        Cev= ((2/math.pi)*math.acos((E1*E2)/(abs(E1)*abs(E2))))
    
    except ValueError: 
        print("Error") 
        Cev=0
    
    return Cev

def adj_finder(matrix, position):
    adj = []
    
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            rangeX = range(0, matrix.shape[0])  # X bounds
            rangeY = range(0, matrix.shape[1])  # Y bounds
            
            (newX, newY) = (position[0]+dx, position[1]+dy)  # adjacent cell
            
            if (newX in rangeX) and (newY in rangeY) and (dx, dy) != (0, 0):
                adj.append((newX, newY))
    
    return adj

def Evidence(img):

    canny= skimage.feature.canny(img).astype(int)*255
    LoG = cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_16S, ksize=3)
    gradient = skimage.filters.sobel(img)*255
    
    img_filtrada= skimage.filters.sobel(img)
    
    R= (canny+LoG+gradient)/(3*255)

   # plt.imshow(canny)
   # plt.imshow(LoG)
   # plt.imshow(gradient)
   # plt.imshow(R)
    Cle=[]
    
    for x in range(len(img)):
        
        row=[]
        
        for y in range(len(img)):
            
            punto=[x,y]
            suma=0
            aux=adj_finder(img, punto)
            N= len(aux)
            
            for i,j in aux:
                    
                suma+= R[i][j]
                      
            row.append(1-(1/N)*suma)
            
        Cle.append(row)

    return Cle, img_filtrada


def GKoller(img):
    
    H=skimage.feature.hessian_matrix(img, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=False)
    w, v = np.linalg.eig(H)
    v=v.T
    CEv=[]
    
    canny= skimage.feature.canny(img)
    
    for i in range(len(img)):

        row=[]
        
        for j in range(len(img)):
            
            p=[i,j]
                
            row.append(Koller([0,0],p,v))
                
        CEv.append(row)
        
    return CEv, canny.astype(int)
       
def Prunning(img):
    
    T=1/14
    kernel=np.array([[1, 1, 1], 
                                  [1, 8, 1], 
                                   [1, 1, 1]])  
    
    kernel= np.multiply(kernel,T)
    filtered = cv2.filter2D(img, -1, kernel)
    binarizado= np.where(filtered >T*1.5,255,0)
    
    return binarizado


def Matriz_Coste(img):
    
    H=skimage.feature.hessian_matrix(img, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=False)
    w, v = np.linalg.eig(H)
    v= v.T
    Cev= Frangi(img)[0]
    Cle= Evidence(img)[0]
    
    C=[]
    w1, w2, w3 = 0.25, 0.55, 0.55
    
    for i in range(1,len(img)):

        row=[]
        
        for j in range(1,len(img)):
            
            p=[i,j]
    
            for q1,q2 in adj_finder(img, p):
                
                row.append(w1*Cev[p[0]][p[1]]+w2*Koller([q1,q2],p,v)+w3*Cle[p[0]][p[1]])
                
        C.append(row)
    
    return C


def algoritmo_segmentacion(img=None):
    
    global imagen_actual
    if img is None: imagen= np.array(imagen_actual)
    else: imagen= img
    t_inicio= time.time()
    print(img)

    luminosidad = imagen
    
    ratio1=3.5
    ratio2=4.5
    ratio3=5.5
    
    w1, w2, w3 = 0.25, 0.55, 0.55
    
    alfa1= cv2.resize(luminosidad, dsize=(int(luminosidad.shape[1]/ratio1),int(luminosidad.shape[0]/ratio1)), interpolation=cv2.INTER_LINEAR)
    alfa2= cv2.resize(luminosidad, dsize=(int(luminosidad.shape[1]/ratio2),int(luminosidad.shape[0]/ratio2)), interpolation=cv2.INTER_LINEAR)
    alfa3= cv2.resize(luminosidad, dsize=(int(luminosidad.shape[1]/ratio3),int(luminosidad.shape[0]/ratio3)), interpolation=cv2.INTER_LINEAR)
    
    f1,e1,k1 = Frangi(alfa1)[1], Evidence(alfa1)[1], Frangi(alfa1)[1]
    
    alfa1= np.add(np.multiply(f1,w1),np.multiply(e1,w3),np.multiply(k1,w2)) 

    f2,e2,k2= Frangi(alfa2)[1], Evidence(alfa2)[1], Frangi(alfa2)[1]
    
    alfa2= np.add(np.multiply(f2,w1),np.multiply(e2,w3),np.multiply(k2,w2)) 
    
    f3,e3,k3= Frangi(alfa3)[1], Evidence(alfa3)[1], Frangi(alfa3)[1]
    
    alfa3= np.add(np.multiply(f3,w1),np.multiply(e3,w3),np.multiply(k3,w2)) 
    
    alfa1= cv2.resize(alfa1,dsize=(luminosidad.shape[1],luminosidad.shape[0]))
    alfa2= cv2.resize(alfa2,dsize=(luminosidad.shape[1],luminosidad.shape[0]))
    alfa3= cv2.resize(alfa3,dsize=(luminosidad.shape[1],luminosidad.shape[0]))
    
    seg= np.stack((alfa1, alfa2, alfa3))
    
    seg= seg.max(0)
    
    final=Prunning(seg)
    print("Duración:", time.time()-t_inicio,"s")    
    imagen_pil = Image.fromarray(final.astype("uint8"))
    
    if img is None:
        mostrar_imagen(imagen_pil)
        imagen_actual= imagen_pil
        imagenes_editadas[indimagen] = imagen_actual
    
    return imagen_pil

def Global_algoritmo_segmentacion():
    
    global imagen_actual
    global imagenes_editadas
    
    for i in range(len(imagenes_editadas)):
        
        img= np.array(imagenes_editadas[i])
        imagenes_editadas[i]= algoritmo_segmentacion(img)
    
    imagen_actual = imagenes_editadas[indimagen]
    mostrar_imagen(imagen_actual)

    return

def LoG():
    
    global imagen_actual
    img= np.array(imagen_actual)
    log = cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_8U, ksize=3)
    print(cv2.cvtColor(cv2.cvtColor(log, cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2HLS).shape)
    mostrar_imagen(array_to_img(cv2.cvtColor(cv2.cvtColor(log, cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2HLS)))
    imagen_actual= array_to_img(cv2.cvtColor(cv2.cvtColor(log, cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2HLS))
    imagenes_editadas[indimagen] = imagen_actual
    return

def GlobalLoG():
    
    global imagen_actual
    global imagenes_editadas
    
    for i in range(len(imagenes_editadas)):
        
        img= np.array(imagenes_editadas[i])
        log = cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_8U, ksize=3)
        imagenes_editadas[i]= array_to_img(cv2.cvtColor(cv2.cvtColor(log, cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2HLS))
    
    imagen_actual = imagenes_editadas[indimagen]
    mostrar_imagen(imagen_actual)

    return

def Gradient():
    
    global imagen_actual
    global imagenes_editadas
    gradient = skimage.filters.sobel(img_to_array(imagen_actual))
    mostrar_imagen(array_to_img(gradient))
    imagen_actual= array_to_img(gradient)
    imagenes_editadas[indimagen] = imagen_actual
    return

def GlobalGradient():
    
    global imagen_actual
    global imagenes_editadas
    
    for i in range(len(imagenes_editadas)):
        
        gradient = skimage.filters.sobel(img_to_array(imagenes_editadas[i]))
        imagenes_editadas[i] = array_to_img(gradient)
    
    imagen_actual = imagenes_editadas[indimagen]
    mostrar_imagen(imagen_actual)
    
    return

def AngioNet():
    
    t_inicio= time.time()
    model = tf.keras.models.load_model("angioesteroides.keras")
    
    # Abrir la imagen con Pillow
    imagen = imagen_actual
    # Redimensionar la imagen al tamaño esperado por el modelo
    imagen = imagen.resize((512, 512))
    # Convertir a un arreglo NumPy
    imagen_array = np.array(imagen)
    # Agregar una dimensión para representar un lote de tamaño 1
    imagen_array = np.expand_dims(imagen_array, axis=0)
    
    img= model.predict(imagen_array)
    
    tensor = img[0]  # Esto nos deja un tensor de forma [512, 512, 2]

    image_prob = np.max(tensor, axis=-1)  # Esto nos da una imagen de tamaño [512, 512] con la probabilidad más alta por píxel

    # Normalizamos la imagen segmentada para asegurar que los valores estén en el rango de 0 a 255
    min_val = np.min(image_prob)
    max_val = np.max(image_prob)

    # Evitamos la división por 0, si el mínimo y máximo son iguales, simplemente devolvemos la imagen original multiplicada por 255
    if min_val != max_val:
        image_normalized = (image_prob - min_val) / (max_val - min_val) * 255
    else:
        image_normalized = image_prob * 255  # Si no hay variación, multiplicamos por 255 para visibilidad
    
    imagen_cv = (image_normalized * 255).astype('uint8')
    imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_GRAY2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)
    mostrar_imagen(imagen_pil)
    print("Duración:", time.time()-t_inicio,"s")

    return

def GlobalAngioNet():
    
    return

def CNNclasica():
    
    print("Realizando predicción con CNN clásica")
    # Cargar el modelo guardado en archivo.h5
    modelo = tf.keras.models.load_model('modelo_cnn.h5')
    
     # Abrir la imagen con Pillow
    imagen = imagen_actual.convert('RGB')  # Convertir a RGB si es necesario
    # Redimensionar la imagen al tamaño esperado por el modelo
    imagen = imagen.resize((224, 224))
    # Convertir a un arreglo NumPy
    imagen_array = np.array(imagen)
    # Escalar los valores de píxeles al rango [0, 1] (opcional, según el modelo)
    imagen_array = imagen_array / 255.0
    # Agregar una dimensión para representar un lote de tamaño 1
    imagen_array = np.expand_dims(imagen_array, axis=0)
    
    # Realizar predicción
    prediccion = modelo.predict(imagen_array)
    resultado = round(prediccion[0][0],2)
    if(resultado>=0.5):
        etiqueta_prediccion.config(text=f"Predicción: lesion\nConfianza: {resultado*100:.2f}%")
        etiqueta_prediccion.pack()  # Asegurarse de mostrar la etiqueta después de la primera predicción
    else:
        etiqueta_prediccion.config(text=f"Predicción: nonlesion\nConfianza: {(1-resultado)*100:.2f}%")
        etiqueta_prediccion.pack()  # Asegurarse de mostrar la etiqueta después de la primera predicción
    
    return resultado

def GlobalCNN():
    
    global resultados_globales
    t_inicio= time.time()
    resultados_globales = []
    modelo = YOLO("runs/classify/train2/weights/best.pt")
    print("Se van a clasificar", len(imagenes_originales),"imágenes")

    for imagen in imagenes_originales:
    
         # Abrir la imagen con Pillow
        imagen = imagen.convert('RGB')  # Convertir a RGB si es necesario
        # Redimensionar la imagen al tamaño esperado por el modelo
        imagen = imagen.resize((224, 224))
        # Convertir a un arreglo NumPy
        imagen_array = np.array(imagen)
        # Escalar los valores de píxeles al rango [0, 1] (opcional, según el modelo)
        imagen_array = imagen_array / 255.0
        # Agregar una dimensión para representar un lote de tamaño 1
        imagen_array = np.expand_dims(imagen_array, axis=0)
        
        # Realizar predicción
        prediccion = modelo.predict(imagen_array)
        resultado = round(prediccion[0][0],2)
        if(resultado>=0.5):
            etiqueta_prediccion.config(text=f"Predicción: lesion\nConfianza: {resultado*100:.2f}%")
            etiqueta_prediccion.pack()
            resultados_globales.append([imagen.filename.split('\\')[-1],"lesion",resultado])
        else:
            etiqueta_prediccion.config(text=f"Predicción: nonlesion\nConfianza: {(1-resultado)*100:.2f}%")
            etiqueta_prediccion.pack()
            resultados_globales.append([imagen.filename.split('\\')[-1],"nonlesion",1-resultado])
    
    print("Duración:", time.time()-t_inicio,"s")
    print("Duración promedio:", (time.time()-t_inicio)/len(imagenes_originales),"s")
    
    return

def YOLOv11():
    
    t_inicio= time.time()
    print("Realizando predicción con YOLO")
    modelo = YOLO("yolo_classify.pt")

    # Asumimos que se usa la imagen editada actual
    resultado = modelo(source=imagen_actual, show=False, save=False, verbose=False)

    probs = resultado[0].probs
    class_index = probs.top1
    class_name = resultado[0].names[class_index]
    score = float(probs.top1conf.cpu().numpy())

    # Mostrar la predicción en la interfaz
    etiqueta_prediccion.config(text=f"Predicción: {class_name}\nConfianza: {score*100:.2f}%")
    etiqueta_prediccion.pack()  # Asegurarse de mostrar la etiqueta después de la primera predicción
    print("Duración:", time.time()-t_inicio,"s")
    
    narrar_en_hilo(f"Predicción: {class_name}\nConfianza: {score*100:.2f}%")
    
    return

def GlobalYOLO():
    
    global resultados_globales
    t_inicio= time.time()
    resultados_globales = []
    modelo = YOLO("yolo_classify.pt")
    print("Se van a clasificar", len(imagenes_originales),"imágenes")

    for imagen in imagenes_originales:
    
        resultado = modelo(source=imagen, show=False, save=False, verbose=False)
        probs = resultado[0].probs
        class_name = resultado[0].names[probs.top1]
        score = float(probs.top1conf.cpu().numpy())
        resultados_globales.append([imagen.filename.split('\\')[-1],class_name,score])
    
    print("Duración:", time.time()-t_inicio,"s")
    print("Duración promedio:", (time.time()-t_inicio)/len(imagenes_originales),"s")
    
    return

def YOLOv11_detect():
    
    global imagen_actual
    global imagenes_editadas
    
    t_inicio= time.time()
    print("Realizando predicción con YOLO")
    modelo = YOLO("yolo_detect.pt")

    # Asumimos que se usa la imagen editada actual
    resultado = modelo(source=imagen_actual, show=False, save=False, verbose=False)
    
    cajas= resultado[0].boxes.xywh
    clases= {0:"0_20", 1:"20_50", 2:"50_70", 3:"70_90", 4:"90_98", 5:"99", 6:"100"}
    
    for caja in cajas:
        
        x, y, w, h= caja[0].item(), caja[1].item(), caja[2].item(), caja[3].item()
        # Crea una capa de dibujo sobre la imagen
        draw = ImageDraw.Draw(imagen_actual)
        
        text= clases.get(int(resultado[0].boxes.cls[0].item()))
        bbox = draw.textbbox((0, 0), text)

        # Calcular las coordenadas para centrar el texto
        text_width = bbox[2] - bbox[0]  # Ancho del cuadro delimitador
        text_height = bbox[3] - bbox[1]  # Alto del cuadro delimitador
        # Resalta la caja con un borde (puedes personalizar el color y grosor)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
        draw.text((x + (w-text_width)/2, y + (h-text_height)/2), text)
    
    try:
        
        imagen_actual= draw._image
        imagenes_editadas[indimagen] = imagen_actual
        mostrar_imagen(imagen_actual)
        score=resultado[0].boxes.conf[0].item()
        # Mostrar la predicción en la interfaz
        etiqueta_prediccion.config(text=f"Confianza: {score*100:.2f}%")
        etiqueta_prediccion.pack()  # Asegurarse de mostrar la etiqueta después de la primera predicción
        print("Duración:", time.time()-t_inicio,"s")
        
        narrar_en_hilo(f"Confianza: {score*100:.2f}%")
        
    except: UnboundLocalError()
        
        
    return

def GlobalYOLO_detect():
    
    global imagen_actual
    global imagenes_editadas
    global resultados_globales
    resultados_globales = []
    
    t_inicio= time.time()
    print("Realizando predicción con YOLO")
    modelo = YOLO("yolo_detect.pt")

    clases= {0:"0_20", 1:"20_50", 2:"50_70", 3:"70_90", 4:"90_98", 5:"99", 6:"100"}
    
    for i in range(len(imagenes_editadas)):
        
        # Asumimos que se usa la imagen editada actual
        resultado = modelo(source=imagenes_editadas[i], show=False, save=False, verbose=False)
        
        cajas= resultado[0].boxes.xywh
        
        for caja in cajas:
            
            x, y, w, h= caja[0].item(), caja[1].item(), caja[2].item(), caja[3].item()
            # Crea una capa de dibujo sobre la imagen
            draw = ImageDraw.Draw(imagenes_editadas[i])
            
            text= clases.get(int(resultado[0].boxes.cls[0].item()))
            bbox = draw.textbbox((0, 0), text)
    
            # Calcular las coordenadas para centrar el texto
            text_width = bbox[2] - bbox[0]  # Ancho del cuadro delimitador
            text_height = bbox[3] - bbox[1]  # Alto del cuadro delimitador
            # Resalta la caja con un borde (puedes personalizar el color y grosor)
            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
            draw.text((x + (w-text_width)/2, y + (h-text_height)/2), text)
            resultados_globales.append([filenames[i],resultado[0].boxes.conf[0].item(),clases.get(int(resultado[0].boxes.cls[0].item())),x,y,w,h])
    
        try:imagenes_editadas[i] = draw._image
        except:UnboundLocalError()

    imagen_actual = imagenes_editadas[indimagen]
    mostrar_imagen(imagen_actual)
    print("Duración:", time.time()-t_inicio,"s")
    
    return

def SVM():
    
    model=joblib.load("svm.pkl")
    imagen_rgb = cv2.cvtColor(np.array(imagen_actual), cv2.COLOR_GRAY2RGB)
    img_resized=cv2.resize(imagen_rgb,(150,150))
    flat_data=np.array(img_resized.flatten())
    
    y_pred = model.predict([np.array(flat_data).ravel()])

    if y_pred==[1]:
        
        etiqueta_prediccion.config(text=f"Predicción: nonlesion\nConfianza: Undefined")
        etiqueta_prediccion.pack()  # Asegurarse de mostrar la etiqueta después de la primera predicción
        
    else:
        
        etiqueta_prediccion.config(text=f"Predicción: lesion\nConfianza: Undefined")
        etiqueta_prediccion.pack()  
        
    print(y_pred)
    
    return

def GlobalSVM():
    
    global resultados_globales
    t_inicio= time.time()
    resultados_globales = []
    modelo=joblib.load("svm.pkl")

    print("Se van a clasificar", len(imagenes_originales),"imágenes")    

    for imagen in imagenes_originales:
    
        imagen_rgb = cv2.cvtColor(np.array(imagen), cv2.COLOR_GRAY2RGB)
        img_resized=cv2.resize(imagen_rgb,(150,150))
        flat_data=np.array(img_resized.flatten())
        
        y_pred = modelo.predict([np.array(flat_data).ravel()])
        
        if y_pred==[1]:
            
            etiqueta_prediccion.config(text=f"Predicción: nonlesion\nConfianza: Undefined")
            etiqueta_prediccion.pack()  # Asegurarse de mostrar la etiqueta después de la primera predicción
            resultados_globales.append([imagen.filename.split('\\')[-1],"nonlesion","undefined"])
            
        else:
            
            etiqueta_prediccion.config(text=f"Predicción: lesion\nConfianza: Undefined")
            etiqueta_prediccion.pack()  
            resultados_globales.append([imagen.filename.split('\\')[-1],"nonlesion","undefined"])
    
    print("Duración:", time.time()-t_inicio,"s")
    print("Duración promedio por imagen:", (time.time()-t_inicio)/len(imagenes_originales),"s")
    
    return
    
def exportar_imagen():
    
    global imagen_actual
    # Pedir al usuario que elija dónde guardar la imagen
    archivo_guardado = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Archivos PNG", "*.png"), ("Archivos JPG", "*.jpg") ,("Todos los archivos", "*.*")])
    
    if archivo_guardado:

        imagen_actual.save(archivo_guardado)
        messagebox.showinfo("Éxito", f"Imagen guardada correctamente en: {archivo_guardado}")

def exportar_resultados():
    
    global resultados_globales
    
    ruta = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Archivos CSV", "*.csv")], title="Guardar como CSV")
    with open(ruta, 'w', newline='') as file:

        writer = csv.writer(file)
        field = ["nombre","diagnostico","confianza"]
        writer.writerow(field)
        
        for resultado in resultados_globales:
            
            writer.writerow(resultado)
        
    return


def exportar_BB():
    
    global resultados_globales
    
    ruta = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Archivos CSV", "*.csv")], title="Guardar como CSV")
    with open(ruta, 'w', newline='') as file:

        writer = csv.writer(file)
        field = ["nombre","confianza","clase","x","y","w","h"]
        writer.writerow(field)
        
        for resultado in resultados_globales:
            
            writer.writerow(resultado)
        
    return


def exportar_imagenes():
    
    global imagenes_editadas
    archivo_zip = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Archivos ZIP", "*.zip")], title="Guardar como ZIP")

    with zipfile.ZipFile(archivo_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
            for img in imagenes_editadas:
                
                nombre = filenames[imagenes_editadas.index(img)]
                # Abrir la imagen con Pillow
                formato=nombre.split(".")[-1]
                # Guardar la imagen como un objeto en memoria (BytesIO)
                img_bytes = BytesIO()
                img.save(img_bytes, format=formato)
                img_bytes.seek(0)
    
                # Agregar la imagen al archivo ZIP
                zipf.writestr(nombre, img_bytes.read())
    
    print(f"Se ha creado el archivo ZIP: {archivo_zip}")    
    return

def reiniciar():
    
    global imagenes_editadas
    global imagen_actual
    imagen_actual = imagenes_originales[indimagen].copy()
    imagenes_editadas[indimagen]=imagen_actual
    mostrar_imagen(imagen_actual)
     
def reiniciar_global():

    global imagenes_editadas
    global imagen_actual
    imagenes_editadas= imagenes_originales.copy()    
    imagen_actual = imagenes_originales[indimagen].copy()
    mostrar_imagen(imagen_actual)
    return

def anterior_imagen():
    global imagen_actual
    global indimagen
    if indimagen > 0:
        indimagen -= 1
        mostrar_imagen(imagenes_editadas[indimagen])
        # Restaurar la imagen editada si se desea aplicar un filtro
        imagen_actual = imagenes_editadas[indimagen].copy()
        # Actualizar la predicción de la nueva imagen
        etiqueta_prediccion.config(text="Predicción: Ninguna\nConfianza: 0%")

def siguiente_imagen():
    global imagen_actual
    global indimagen
    if indimagen < len(imagenes_editadas) - 1:
        indimagen += 1
        mostrar_imagen(imagenes_editadas[indimagen])
        # Restaurar la imagen editada si se desea aplicar un filtro
        imagen_actual = imagenes_editadas[indimagen].copy()
        # Actualizar la predicción de la nueva imagen
        etiqueta_prediccion.config(text="Predicción: Ninguna\nConfianza: 0%")

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("AngioPixel")
ventana.geometry("305x225")

# Crear un botón para abrir el cuadro de diálogo de selección de archivo
btn_seleccionar = tk.Button(ventana, text="Seleccionar Imagen", command=seleccionar_archivo)
btn_seleccionar.pack(pady=10)

# Crear una etiqueta para mostrar la imagen
etiqueta_imagen = tk.Label(ventana)
etiqueta_imagen.pack(pady=10)

# Crear botones de navegación entre imágenes
btn_anterior = tk.Button(ventana, text="Anterior", command=anterior_imagen, state=tk.DISABLED)
btn_anterior.pack(side=tk.LEFT, padx=10)

btn_siguiente = tk.Button(ventana, text="Siguiente", command=siguiente_imagen, state=tk.DISABLED)
btn_siguiente.pack(side=tk.RIGHT, padx=10)

# Crear un botón para exportar la imagen
btn_reiniciar = tk.Button(ventana, text="Reiniciar", command=reiniciar, state=tk.DISABLED)
btn_reiniciar.pack(pady=10)

# Crear una etiqueta para mostrar la predicción de YOLO
etiqueta_prediccion = tk.Label(ventana, text="Predicción: Ninguna\nConfianza: 0%", font=("Helvetica", 12))
etiqueta_prediccion.pack_forget()  # Inicialmente la ocultamos

# Crear el menú de opciones
menu_bar = tk.Menu(ventana)

# Menú de "Archivos"
menu_archivos = tk.Menu(menu_bar, tearoff=0)
menu_archivos.add_command(label="Abrir archivo", command=lambda: seleccionar_archivo())
menu_archivos.add_command(label="Guardar archivo", command=lambda: exportar_imagen(), state=tk.DISABLED)
menu_archivos.add_command(label="Exportar imágenes", command=lambda: exportar_imagenes(), state=tk.DISABLED)
menu_archivos.add_command(label="Exportar resultados", command=lambda: exportar_resultados(), state=tk.DISABLED)
menu_archivos.add_command(label="Exportar bounding boxes", command=lambda: exportar_BB(), state=tk.DISABLED)
menu_archivos.add_command(label="Reiniciar imágenes", command=lambda: reiniciar_global(), state=tk.DISABLED)
menu_bar.add_cascade(label="Archivos", menu=menu_archivos)

# Menú de "Segmentación"
menu_segmentacion = tk.Menu(menu_bar, tearoff=0)
menu_segmentacion.add_command(label="Algorítmico", command=lambda: algoritmo_segmentacion(), state=tk.DISABLED)
menu_segmentacion.add_command(label="Algorítmico Global", command=lambda: Global_algoritmo_segmentacion(), state=tk.DISABLED)
menu_segmentacion.add_command(label="Laplacian of Gaussian", command=lambda: LoG(), state=tk.DISABLED)
menu_segmentacion.add_command(label="Laplacian of Gaussian Global", command=lambda: GlobalLoG(), state=tk.DISABLED)
menu_segmentacion.add_command(label="Gradiente", command=lambda: Gradient(), state=tk.DISABLED)
menu_segmentacion.add_command(label="Gradiente Global", command=lambda: GlobalGradient(), state=tk.DISABLED)
menu_segmentacion.add_command(label="AngioNet", command=lambda: AngioNet(), state=tk.DISABLED)
menu_segmentacion.add_command(label="AngioNet Global", command=lambda: GlobalAngioNet(), state=tk.DISABLED)
menu_bar.add_cascade(label="Segmentación", menu=menu_segmentacion)

# Menú de "Clasificación"
menu_clasificacion = tk.Menu(menu_bar, tearoff=0)
menu_clasificacion.add_command(label="CNN Clásica", command=lambda: CNNclasica(), state=tk.DISABLED)
menu_clasificacion.add_command(label="CNN Global", command=lambda: GlobalCNN(), state=tk.DISABLED)
menu_clasificacion.add_command(label="YOLOv11", command=lambda: YOLOv11(), state=tk.DISABLED)
menu_clasificacion.add_command(label="YOLOv11 Global", command=lambda: GlobalYOLO(), state=tk.DISABLED)
menu_clasificacion.add_command(label="SVM", command=lambda: SVM(), state=tk.DISABLED)
menu_clasificacion.add_command(label="SVM Global", command=lambda: GlobalSVM(), state=tk.DISABLED)
menu_bar.add_cascade(label="Clasificación", menu=menu_clasificacion)

# Menú de "Detección"
menu_deteccion = tk.Menu(menu_bar, tearoff=0)
menu_deteccion.add_command(label="YOLOv11", command=lambda: YOLOv11_detect(), state=tk.DISABLED)
menu_deteccion.add_command(label="YOLOv11 Global", command=lambda: GlobalYOLO_detect(), state=tk.DISABLED)
menu_bar.add_cascade(label="Detección", menu=menu_deteccion)

speaker = win32com.client.Dispatch("SAPI.SpVoice") 
# Asignar el menú a la ventana
ventana.config(menu=menu_bar)

# Iniciar la aplicación
ventana.mainloop()
