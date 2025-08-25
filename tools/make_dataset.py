import argparse
from pathlib import Path
import shutil
import glob
import numpy as np
import cv2
import os
from tqdm import tqdm
import supervision as sv

def convert_masks_to_polygon(folder: Path):
    mask_files = [file for file in folder.glob('*.png') if file.name.endswith('watershed_mask.png')]
    for mask_file in tqdm(mask_files, desc='Converting masks to polygons', total=len(mask_files), unit='masks'):
        label_file = str(mask_file).replace('endo_watershed_mask.png', 'endo.txt')
        with open(label_file, 'w') as f:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            width = mask.shape[1]
            height = mask.shape[0]
            polygons = sv.mask_to_polygons(mask)
            for polygon in polygons:
                f.write("0 ")
                for point in polygon:
                    f.write(f"{point[0] / width} {point[1] / height} ")
                f.write("\n")
        os.remove(mask_file)


def convert_watershed_to_binary(folder: Path):
    image_files = list(folder.glob('*.png'))
    for image_file in tqdm(image_files, desc="Converting watershed images to binary images", unit="images", total=len(image_files)):
        if "watershed_mask" in image_file.name:
            img = cv2.imread(str(image_file))
            img = np.where(img == (31, 31, 31), 255, img)
            img = np.where(img == (32, 32, 32), 255, img)
            img = np.where(img == (255, 255, 255), 255, 0)
            cv2.imwrite(str(image_file), img)

def remove_other_masks(folder: Path):
    files = list(folder.glob('*.png'))
    for file in tqdm(files, desc="Removing other masks", unit="files", total=len(files)):
        if file.name.endswith('endo_color_mask.png') or file.name.endswith('endo_mask.png'):
            os.remove(str(file))

def copy_rename(img_file, dest, ctr, is_img=True):
    if is_img:
        shutil.copy2(img_file, f'{dest}/{ctr}.jpg')
    else:
        shutil.copy2(img_file, f'{dest}/{ctr}.txt')

def combine_images(source_dir, dest_dir, dataset):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset.lower()
    image_ctr = 0
    if dataset == "flim":
        image_files = list(source_dir.rglob("*.jpg"))
        for image_file in tqdm(image_files, unit="images"):
            shutil.copy2(image_file, str(dest_dir))
            json_file = str(image_file).replace('.jpg', '.json')
            shutil.copy2(json_file, str(dest_dir))
            image_ctr += 1
    elif dataset == "cholec":
        train_val_folders = list(source_dir.glob("*"))
        for train_val_folder in train_val_folders:
            videos = list(train_val_folder.glob("*"))
            for video in tqdm(videos, desc=f"Combining videos in {train_val_folder.name}", unit="folders", total=len(videos)):
                images = video.rglob("*.png")
                for image in images:
                    new_name = train_val_folder.name + "_" + video.name + "_" + image.name
                    new_path = dest_dir / new_name
                    shutil.copy2(image, str(new_path))
                    image_ctr += 1


def create_test_train_split(source_dir: Path, dest_dir: Path, train_split: int, test_split: int, dataset: str):
    train_split = train_split / 100
    test_split = test_split / 100
    val_split = 1 - train_split - test_split
    num_images = len(list(source_dir.glob("*.jpg"))) if dataset == "flim" else len(list(source_dir.glob("*.png")))
    num_train = int(num_images * train_split)
    num_val = int(num_images * val_split)
    num_test = num_images - num_train - num_val if test_split != 0 else 0

    Path(f'{dest_dir}/images/train').mkdir(parents=True, exist_ok=True)
    Path(f'{dest_dir}/images/val').mkdir(parents=True, exist_ok=True)
    Path(f'{dest_dir}/images/test').mkdir(parents=True, exist_ok=True)
    Path(f'{dest_dir}/labels/train').mkdir(parents=True, exist_ok=True)
    Path(f'{dest_dir}/labels/val').mkdir(parents=True, exist_ok=True)
    Path(f'{dest_dir}/labels/test').mkdir(parents=True, exist_ok=True)

    print("Creating Dataset Split")
    if dataset == "flim":
        print(f'Train split: {round(train_split * 100)}%')
        print(f'Val split: {round(val_split * 100)}%')
        print(f'Test split: {round(test_split * 100)}%')
        print(f'Number of images: {num_images}')
        print(f"Number of training images: {num_train}")
        print(f"Number of validation images: {num_val}")
        print(f"Number of testing images: {num_test}")

    if dataset == "flim":
        img_files = sorted(glob.glob(f'{source_dir}/*.jpg'))
        for idx, img_file in enumerate(img_files):
            if idx + 1 < num_train:
                copy_rename(img_file, f'{dest_dir}/images/train', idx)
                copy_rename(img_file.replace('.jpg', '.txt'), f'{dest_dir}/labels/train', idx, False)
            elif idx + 1 < num_val + num_train or num_test == 0:
                copy_rename(img_file, f'{dest_dir}/images/val', idx)
                copy_rename(img_file.replace('.jpg', '.txt'), f'{dest_dir}/labels/val', idx, False)
            else:
                copy_rename(img_file, f'{dest_dir}/images/test', idx)
                copy_rename(img_file.replace('.jpg', '.txt'), f'{dest_dir}/labels/test', idx, False)
    elif dataset == "cholec":
        file_ctr = 0
        image_files = list(source_dir.glob("*.png"))
        for image_file in tqdm(image_files, desc=f"Creating {dataset} dataset split", unit="images", total=len(image_files)):
            label_file = str(image_file).replace('.png', '.txt')
            if image_file.name.startswith("train"):
                copy_rename(image_file, f'{dest_dir}/images/train', file_ctr)
                copy_rename(label_file, f'{dest_dir}/labels/train', file_ctr, is_img=False)
            elif image_file.name.startswith("validation"):
                copy_rename(image_file, f'{dest_dir}/images/val', file_ctr)
                copy_rename(label_file, f'{dest_dir}/labels/val', file_ctr, is_img=False)
            file_ctr += 1
            os.remove(str(image_file))
            os.remove(str(label_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory containing the images and annotations.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory where to save the images.", required=True)
    parser.add_argument("--dataset", type=str, help="Source image name.", default="flim")
    parser.add_argument('--train_split', type=int, default=70)
    parser.add_argument('--test_split', type=int, default=0)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating Yolo dataset for {args.dataset}.")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    combine_images(input_dir, output_dir, dataset=args.dataset)
    if args.dataset.lower() == "cholec":
        convert_watershed_to_binary(output_dir)
        remove_other_masks(output_dir)
        convert_masks_to_polygon(output_dir)
    create_test_train_split(output_dir, output_dir, args.train_split, args.test_split, args.dataset)


