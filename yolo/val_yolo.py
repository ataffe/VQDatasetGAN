from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/segment/train/weights/best.pt')
    metrics = model.val()
    print(f'mAP 50 - 95: {metrics.seg.map}')
    print(f'mAP 50: {metrics.seg.map50}')
    print(f'mAP 75: {metrics.seg.map75}')
    print(f'mAP categories: {metrics.seg.maps}')