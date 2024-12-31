from ultralytics import YOLO

#modelo = YOLO("yolo11n.pt")
modelo = YOLO("runs/detect/train4/weights/best.pt")

#results = modelo.train(data="dataset_detect/config.yaml",epochs=40, save=True)
resultado = modelo(source="dataset/val/lesion/p9_v5_00042.png", show=True, conf=0.1, save=False, verbose=False)

ojo= modelo.val(data="dataset_detect/config.yaml", save=True)

cajas=resultado[0].boxes
clases= cajas.cls


print(clases)
