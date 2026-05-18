from ultralytics import YOLO

def main():
    model = YOLO('yolo26n.pt')

    model.train(
        data=r'data.yaml',
        epochs=350,
        imgsz=640,
        batch=8, # Pour 4070, 8 pour 3060
        device=0,
        workers=2,
        cache="ram", # "disk" or False if not enough dedicated RWAM

        patience=50,
        optimizer="MuSGD",

        momentum  = 0.937,
        weight_decay = 0.0005,

        dropout=0.1,
        warmup_epochs=5,

        # ---- DATA AUGMENTATION ----

        fliplr = 0.25,
        flipud = 0.1,

        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.2,

        mosaic = 0.0,

        project=r'runs'
    )


if __name__ == "__main__":
    main()