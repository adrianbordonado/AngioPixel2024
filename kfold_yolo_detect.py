from pathlib import Path
from ultralytics import YOLO
import yaml

dataset_path = Path("C:/Users/adria/Downloads/CADICA/dataset_detect/")  # replace with 'path/to/dataset' for your custom data
labels = sorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'

yaml_file = "dataset_detect/config.yaml"  # your data YAML with data directories and names dictionary
with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = sorted(classes.keys())

import pandas as pd

indx = [label.stem for label in labels]  # uses base filename as ID (no extension)
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

from collections import Counter

for label in labels:
    lbl_counter = Counter()

    with open(label, "r") as lf:
        lines = lf.readlines()

    for line in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(line.split(" ")[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

from sklearn.model_selection import KFold

ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

kfolds = list(kf.split(labels_df))

folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=indx, columns=folds)

for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df[f"split_{idx}"].loc[labels_df.iloc[train].index] = "train"
    folds_df[f"split_{idx}"].loc[labels_df.iloc[val].index] = "val"
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio
    
import datetime

supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images = []

# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

# Create the necessary directories and dataset YAML files (unchanged)
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (split_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "names": classes,
            },
            ds_y,
        )

import shutil

for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / "images" / k_split
        lbl_to_path = save_path / split / "labels" / k_split

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)        
        
results = {}
model = YOLO("runs/detect/train4/weights/best.pt")
# Define your additional arguments here
batch = 16
project = "kfold_demo"
epochs = 1
precision=0
recall=0
mAP50=0
mAP5095=0

for k in range(ksplit):
    dataset_yaml = ds_yamls[k]
    model.val(data=dataset_yaml,verbose=False, save=False)  # include any train arguments
    results[k] = model.metrics  # save output metrics for further analysis
    
for i in range(len(results)):
    
    resultado= results[i].results_dict
    
    precision+= resultado.get("metrics/precision(B)")
    recall+= resultado.get("metrics/recall(B)")
    mAP50+= resultado.get("metrics/mAP50(B)")
    mAP5095+= resultado.get("metrics/mAP50-95(B)")

precision= precision/len(results)
recall= recall/len(results)    
mAP50= mAP50/len(results)
mAP5095= mAP5095/len(results)

f1scores= results[0].box.f1
iou= f1scores[0]/(2-f1scores[0])