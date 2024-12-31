from ultralytics import YOLO
import sys
import io

# Configurar salida estándar en UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ruta de la imagen
ruta = sys.argv[1]
ruta_imagen = '' + ruta

#modelo = YOLO("yolo11x-cls.pt")
modelo = YOLO("best.pt")

#results = modelo.train(data="C:/Users/adria/Downloads/CADICA/dataset",epochs=10, save=True)
resultado = modelo(source=ruta_imagen, show=False, conf=0.97, save=False, verbose=False)

probs = resultado[0].probs
class_index = probs.top1
class_name = resultado[0].names[class_index]
score = float(probs.top1conf.cpu().numpy())
score = score*100

print("Modelo: yolo")
if(class_name == "nonlesion"):
    print("Predicción: No tiene lesión")
else:
    print("Predicción: Tiene lesión")

print(f"Confianza: {score:.2f}%")
