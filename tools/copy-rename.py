import argparse
from pathlib import Path
import shutil

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=Path, required=True)
    parser.add_argument('--output_folder', type=Path, required=True)
    parser.add_argument('--name-prefix', type=str, required=False)
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    name_prefix = args.name_prefix
    img_files = list(input_folder.rglob('*.jpg'))
    for img_file in tqdm(img_files, total=len(img_files), desc='Copying images', unit='images'):
        new_file_name = f'{name_prefix}-{img_file.name}'
        shutil.copy(img_file, output_folder / new_file_name)