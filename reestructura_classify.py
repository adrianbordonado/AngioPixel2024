import shutil
import os
import random

train_split = 80
val_split = 20
test_split = 20
directorio = "selectedVideos"

#REQUIERE CREAR ESTRUCTURA DE CARPETAS PREVIAMENTE!!!

for paciente in os.listdir(directorio):
    
    lista_lesionados = open(directorio+"/"+paciente+"/lesionVideos.txt","r")
    lista_no_lesionados = open(directorio+"/"+paciente+"/nonlesionVideos.txt","r")
    
    for video in lista_lesionados:
    
        video= video[:-1]
        
        for imagen in os.listdir(directorio+"/"+paciente+"/"+video+"/input"):
            
            if random.choices(["train","test"], weights=(train_split,test_split))[0] == "train":
                
                if random.choices(["train","val"], weights=(train_split,val_split))[0] == "train":
    
                    shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"train/lesion/"+imagen)
        
                else:
                    
                    shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"val/lesion/"+imagen)
                    
            else:
                
                shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"test/lesion/"+imagen)
            
    for video in lista_no_lesionados:
     
         video= video[:-1]
         
         for imagen in os.listdir(directorio+"/"+paciente+"/"+video+"/input"):
         
             if random.choices(["train","test"], weights=(train_split,test_split))[0] == "train":
                 
                 if random.choices(["train","val"], weights=(train_split,val_split))[0] == "train":
     
                     shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"train/nonlesion/"+imagen)
         
                 else:
                     
                     shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"val/nonlesion/"+imagen)
                     
             else:
                 
                 shutil.copyfile(directorio+"/"+paciente+"/"+video+"/input/"+imagen,"test/nonlesion/"+imagen)
                 
    lista_lesionados.close()
    lista_no_lesionados.close()
        
        

