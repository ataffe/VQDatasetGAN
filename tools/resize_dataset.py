import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
from multiprocessing import Pool

def resize_image(file):
    img = cv2.imread(str(file))
    if img is None:
        print(f"Warning: Could not read image {file}")
        return
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(file), img_resized)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, help="Path to the source dataset directory")
    args = parser.parse_args()
    source_dir = args.dir
    print(f"Source directory: {source_dir}")
    files = list(source_dir.glob("*.jpg"))
    with Pool(10) as pool:
        r = list(tqdm(pool.imap(resize_image, files), total=len(files), desc="Resizing images", unit="images"))
    print(f'Resized {len(r)} images to 256x256 pixels.')