import random
from ultralytics import YOLO

def main():
    model = YOLO('yolo26s.pt')
    seed = random.randint(0, 9999)

    model.train(
        data=r'data.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        cache="ram",
        save_period=50,
        patience=50,
        optimizer="AdamW",
        momentum=0.937,
        weight_decay=0.0005,
        dropout=0.1,
        warmup_epochs=5,
        # ---- DATA AUGMENTATION ----
        fliplr=0.25,
        flipud=0.1,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        mosaic=1.0,
        auto_augment="randaugment",
        seed=seed,
        project=r'runs'
    )

if __name__ == "__main__":
    main()

# BASE

# data = r'data.yaml',
# epochs = 350,
# imgsz = 640,
# batch = 8,
# device = 0,
# workers = 2,
# cache = "ram",
# save_period = 50,
#
# patience = 50,
# optimizer = "AdamW",
#
# momentum = 0.937,
# weight_decay = 0.0005,
#
# dropout = 0.1,
# warmup_epochs = 5,
#
# # ---- DATA AUGMENTATION ----
#
# fliplr = 0.25,
# flipud = 0.1,
#
# hsv_h = 0.015,
# hsv_s = 0.35,
# hsv_v = 0.2,
#
# mosaic = 0.0,
#
# seed = seed,
#
# project = r'runs'

# ULTRALITICS
#
# data=r'data.yaml',
# device = 0,
# mosaic=0.5,
# mixup=0.0,
# copy_paste=0.0,
# lr0=0.001,
# epochs=50,
# freeze=10,
# seed=seed,
# project=r'runs'
