from ultralytics import YOLO

#modelo = YOLO("yolo11x-cls.pt")
modelo = YOLO("runs/classify/train2/weights/best.pt")

#results = modelo.train(data="C:/Users/adria/Downloads/CADICA/dataset",epochs=10, save=True)
resultado = modelo(source="dataset/train/lesion/p9_v1_00001.png", show=False, conf=0.97, save=False, verbose=False)

probs = resultado[0].probs
class_index = probs.top1
class_name = resultado[0].names[class_index]
score = float(probs.top1conf.cpu().numpy())

print(class_name, score)