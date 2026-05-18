import os
import random
import shutil

base_dir = "dataset"
out_dir = "dataset_yolo"

images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Création des dossiers train / val / test
for split in ["train", "val", "test"]:
    os.makedirs(f"{out_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{out_dir}/labels/{split}", exist_ok=True)

# Récupération des images
images = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# Mélange aléatoire
random.shuffle(images)

# Calcul des splits 80 / 10 / 10
total = len(images)
print(total)

train_end = int(total * 0.8)
val_end = train_end + int(total * 0.1)

train = images[:train_end]
val = images[train_end:val_end]
test = images[val_end:]

def move(data, mode):
    for img in data:
        img_path = os.path.join(images_dir, img)

        label_name = img.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        # Copie image
        shutil.copy(
            img_path,
            os.path.join(out_dir, "images", mode, img)
        )

        # Copie label si présent
        if os.path.exists(label_path):
            shutil.copy(
                label_path,
                os.path.join(out_dir, "labels", mode, label_name)
            )

# Répartition des données
move(train, "train")
move(val, "val")
move(test, "test")

print("Dataset prêt ✔")
print(f"Train : {len(train)} images")
print(f"Validation : {len(val)} images")
print(f"Test : {len(test)} images")