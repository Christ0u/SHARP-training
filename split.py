import os
import random
import shutil

base_dir = "dataset"
out_dir = "dataset_yolo"

images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

os.makedirs(f"{out_dir}/images/train", exist_ok=True)
os.makedirs(f"{out_dir}/images/val", exist_ok=True)
os.makedirs(f"{out_dir}/labels/train", exist_ok=True)
os.makedirs(f"{out_dir}/labels/val", exist_ok=True)

images = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

split = int(len(images) * 0.8)
train = images[:split]
val = images[split:]

def move(data, mode):
    for img in data:
        img_path = os.path.join(images_dir, img)
        label_path = os.path.join(labels_dir, img.rsplit(".",1)[0] + ".txt")

        shutil.copy(img_path, f"{out_dir}/images/{mode}/{img}")

        if os.path.exists(label_path):
            shutil.copy(label_path, f"{out_dir}/labels/{mode}/{img.rsplit('.',1)[0]}.txt")

move(train, "train")
move(val, "val")

print("Dataset prêt ✔")