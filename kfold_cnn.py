import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import shutil
from pathlib import Path
import datetime

ksplit=5
# Ruta del modelo preentrenado (ajusta según el lugar donde se encuentra tu modelo)
modelo_path = 'cnn.keras'  # Actualiza la ruta del modelo

# Cargar el modelo preentrenado
model = tf.keras.models.load_model(modelo_path)

# Definir las rutas de los datos
dataset_path = Path("C:/Users/adria/Downloads/CADICA/dataset/")  # Ruta al dataset
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)

# Recolectar las imágenes y etiquetas
supported_extensions = [".jpg", ".jpeg", ".png"]
images_lesion = []
images_nonlesion = []

for ext in supported_extensions:
    images_lesion.extend(sorted((dataset_path / "test/lesion").rglob(f"*{ext}")))
    images_nonlesion.extend(sorted((dataset_path / "test/nonlesion").rglob(f"*{ext}")))

# Combinar imágenes y etiquetas
all_images = images_lesion + images_nonlesion
labels = ["lesion"] * len(images_lesion) + ["nonlesion"] * len(images_nonlesion)

# Configurar K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Crear las carpetas necesarias para los splits
for fold, (train_index, val_index) in enumerate(kf.split(all_images)):
    print(f"Procesando Fold {fold + 1}...")

    # Crear directorios para el fold actual
    train_lesion_dir = save_path / f"fold_{fold + 1}" / "train" / "lesion"
    val_lesion_dir = save_path / f"fold_{fold + 1}" / "val" / "lesion"
    train_nonlesion_dir = save_path / f"fold_{fold + 1}" / "train" / "nonlesion"
    val_nonlesion_dir = save_path / f"fold_{fold + 1}" / "val" / "nonlesion"
    
    for directory in [train_lesion_dir, val_lesion_dir, train_nonlesion_dir, val_nonlesion_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copiar imágenes a las carpetas correspondientes
    for idx in train_index:
        image = all_images[idx]
        label = labels[idx]
        if label == "lesion":
            shutil.copy(image, train_lesion_dir / image.name)
        else:
            shutil.copy(image, train_nonlesion_dir / image.name)

    for idx in val_index:
        image = all_images[idx]
        label = labels[idx]
        if label == "lesion":
            shutil.copy(image, val_lesion_dir / image.name)
        else:
            shutil.copy(image, val_nonlesion_dir / image.name)
    
    print(f"Fold {fold + 1} completado.")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

results = {}
matrices = []

# Generador de imágenes
datagen = ImageDataGenerator(rescale=1./255)

# Validación cruzada (K-Fold)
for fold, (train_index, val_index) in enumerate(kf.split(all_images)):
    print(f"Evaluando Fold {fold + 1}...")

    # Preparar las imágenes de entrenamiento y validación
    X_train = [all_images[i] for i in train_index]
    X_val = [all_images[i] for i in val_index]
    y_train = [labels[i] for i in train_index]
    y_val = [labels[i] for i in val_index]

    # Utilizar ImageDataGenerator para cargar las imágenes
    train_gen = datagen.flow_from_directory(
        save_path / f"fold_{fold + 1}" / "train",
        target_size=(224,224), 
        batch_size=16,
        class_mode="binary",
        subset="training"
    )
    
    val_gen = datagen.flow_from_directory(
        save_path / f"fold_{fold + 1}" / "val",
        target_size=(224,224), 
        batch_size=16,
        class_mode="binary",
    )

    # Evaluar el modelo en el conjunto de validación
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"Fold {fold + 1} - Loss: {val_loss}, Accuracy: {val_acc}")
    
    # Guardar métricas y matrices de confusión
    results[fold] = {"loss": val_loss, "accuracy": val_acc}

    # Obtener las predicciones en el conjunto de validación
    y_pred = model.predict(val_gen, steps=val_gen.samples // val_gen.batch_size, verbose=1)

    # Redondear las predicciones para obtener clases (0 o 1)
    y_pred_classes = (y_pred > 0.5).astype("int32")  # Umbral para clasificación binaria
    
    # Asegurarse de que el número de muestras en y_pred_classes y y_val coincidan
    y_val = np.array(y_val)  # Convertir a array para asegurarse de que sea un array numpy
    print(f"y_pred_classes shape: {y_pred_classes.shape}")
    print(f"y_val shape: {y_val.shape}")

    y_true = val_gen.classes  # Verdaderos valores de las etiquetas
    y_pred = model.predict(val_gen)  # Predicciones del modelo
    
    # Si es clasificación binaria, debes convertir las predicciones a 0 o 1
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calcular las métricas
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")

# Mostrar resultados finales
for fold, metrics in results.items():
    print(f"Fold {fold + 1} - Loss: {metrics['loss']}, Accuracy: {metrics['accuracy']}")

