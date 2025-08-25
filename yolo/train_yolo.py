from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n-seg.pt')
    model.info()
    model.train(data="/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/SSD/SurgicalToolDataset/cholec-seg-8k.yaml", epochs=100, imgsz=640)

