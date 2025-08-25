import argparse
import glob
import shutil
from pathlib import Path

def copy_rename(img_file, dest, ctr, is_img=True):
    if is_img:
        shutil.copy2(img_file, f'{dest}/{ctr}.jpg')
    else:
        shutil.copy2(img_file, f'{dest}/{ctr}.txt')

def create_test_train_split(source_dir, dest_dir, train_split, test_split):
    train_split = train_split / 100
    test_split = test_split / 100
    val_split = 1 - train_split - test_split
    num_images = len(glob.glob(f'{source_dir}/*.jpg'))
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
    print(f'Train split: {round(train_split * 100)}%')
    print(f'Val split: {round(val_split * 100)}%')
    print(f'Test split: {round(test_split * 100)}%')
    print(f'Number of images: {num_images}')
    print(f"Number of training images: {num_train}")
    print(f"Number of validation images: {num_val}")
    print(f"Number of testing images: {num_test}")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--train_split', type=int, default=70)
    parser.add_argument('--test_split', type=int, default=0)
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()
    create_test_train_split(args.source, args.outdir, args.train_split, args.test_split)
    print("Done")
