import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    source_dir = Path("/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/SSD/SurgicalToolDataset/originals/CholecSeg8k")
    output_dir = Path("/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/SSD/SurgicalToolDataset/CholecSeg8k-images")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_val_folders = list(source_dir.glob("*"))
    for train_val_folder in train_val_folders:
        videos = list(train_val_folder.glob("*"))
        for video in tqdm(videos, desc=f"Combining videos in {train_val_folder.name}", unit="folders", total=len(videos)):
            images = video.rglob("*_endo.png")
            for image in images:
                img = cv2.imread(str(image))
                img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)
                new_img_path = output_dir / f"{image.name.replace('.png', '.jpg')}"
                cv2.imwrite(str(new_img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])