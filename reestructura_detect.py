import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from segmentado_multiescala import algoritmo_segmentacion
import random
import shutil

"""
for archivo in os.listdir("entrenamiento_angionet/train"):
    
    if archivo[-3:]=="txt":
        
        gt= open("entrenamiento_angionet/train/"+archivo,"r")
        
        for linea in gt:
            
            mascara = np.array([list(map(int, line.split())) for line in [linea.strip().split('p', 1)[0]]])
            
            x, y, w, h = mascara[0][0], mascara[0][1], mascara[0][2], mascara[0][3]
            
            image = Image.open("entrenamiento_angionet/train/"+archivo[:-3]+'png')
            # Recorta la zona dentro de la caja
            cropped_image = image.crop((x, y, x + w, y + h))
            
            # Crea una capa de dibujo sobre la imagen
            draw = ImageDraw.Draw(image)
            
            # Resalta la caja con un borde (puedes personalizar el color y grosor)
            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
            image.save("entrenamiento_angionet/masks/"+archivo[:-3]+"png")

"""
"""
for archivo in os.listdir("entrenamientoangionet/input/train"):
    
    image = Image.open("entrenamientoangionet/input/train/"+archivo[:-3]+'png')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
    image = algoritmo_segmentacion(np.array(image))
    cv2.imwrite('masks/'+archivo[:-3]+"png", image)
"""
"""
for archivo in os.listdir("masks/target"):
    count+=1
    image = Image.open("masks/target/"+archivo[:-3]+'png')
    image= cv2.resize(np.array(image), (512, 512))
    cv2.imwrite('masks/target/'+archivo[:-3]+"png", image)
"""

train_split = 80
val_split = 20
test_split = 20
directorio = "selectedVideos"
       
clases= {"0_20":0, "20_50":1, "50_70":2, "70_90":3, "90_98":4, "99":5, "100":6}

c1=0; c2=0; c3=0; c4=0; c5=0; c6=0; c7=0;      

#LIMPIAR CARPETAS ANTES!!!

for paciente in os.listdir(directorio):
    
    lista_lesionados = open(directorio+"/"+paciente+"/lesionVideos.txt","r")
    
    for video in lista_lesionados:
    
        video= video[:-1]
                  
        for label in os.listdir(directorio+"/"+paciente+"/"+video+"/groundtruth"):
            
            if (label[-3:]=="txt"):
                
                gt= open(directorio+"/"+paciente+"/"+video+"/groundtruth/"+label,"r")
                texto=[]
                imagen=label[:-3]+"png "
                
                for linea in gt:
                    
                    clase=clases.get(linea.strip().split('p', 1)[1])
                    
                    if (clase==0):c1+=1
                    elif(clase==1):c2+=1
                    elif(clase==2):c3+=1
                    elif(clase==3):c4+=1
                    elif(clase==4):c5+=1
                    elif(clase==5):c6+=1
                    elif(clase==6):c7+=1
                    
                    caja = linea.strip().split('p', 1)[0]
                    caja= [float(element)/512 for element in caja.split()]
                    caja= [str(element) for element in caja]
                    caja = " ".join(caja)
                    texto.append(str(clase)+" "+caja)
                    
            
                r1=random.choices(["train","test"], weights=(train_split,test_split))[0]
                r2=random.choices(["train","val"], weights=(train_split,val_split))[0]
                if r1 == "train":
                    
                    if r2 == "train":
                        
                        shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"entrenamientoangionet/images/train/"+imagen)
                        with open("entrenamientoangionet/labels/train/"+label, "w") as f:
                            
                            for linea in texto:
                                f.write(linea + "\n")
                    else:
                        
                        shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"entrenamientoangionet/images/val/"+imagen)
                        with open("entrenamientoangionet/labels/val/"+label, "w") as f:
                            
                            for linea in texto:
                                f.write(linea + "\n")
                else:
                    
                    shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"entrenamientoangionet/images/test/"+imagen)
                    with open("entrenamientoangionet/labels/test/"+label, "w") as f:
                        
                        for linea in texto:
                            f.write(linea + "\n")
        
        gt.close()

    lista_lesionados.close()

print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
print(c6)
print(c7)