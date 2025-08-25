import os
import glob
import json
from pathlib import Path
import shutil
import argparse

to_class_idx = {
    "AimingBeam": '0',
    "Prob": '1',
    "Instrument": '1'
}

def save_yolo_format(folder, outdir, file_count, files_without):
    json_files = glob.glob(folder + '/*.json')
    for json_file in json_files:
        img_file = json_file.split('/')[-1].replace('.json', '.jpg')
        print(img_file)
        with open(json_file, 'r') as f:
            annotation = json.load(f)
            polygons = []
            width = float(annotation['imageWidth'])
            height = float(annotation['imageHeight'])
            for shape in annotation['shapes']:
                if len(shape['points']) > 2 and shape["label"] in to_class_idx:
                    label = shape['label']
                    polygon = {
                        'label': label,
                        'points': shape['points']
                    }
                    polygons.append(polygon)
            if len(polygons) > 0:
                shutil.copy2(f'{folder}/{img_file}', f'{outdir}/{file_count}.jpg')
                with open(f'{outdir}/{file_count}.txt', 'w') as outfile:
                    for polygon in polygons:
                        outfile.write(to_class_idx[polygon['label']])
                        for point in polygon['points']:
                            outfile.write(f' {point[0] / width} {point[1] / height}')
                        outfile.write('\n')
                file_count += 1
            else:
                print(f'no annotation for {img_file}')
                files_without += 1
    return file_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    source = args.source
    outdir = args.outdir

    Path(outdir).mkdir(parents=True, exist_ok=True)
    file_count = 0
    files_without = 0
    folders = [name for name in os.listdir(source) if os.path.isdir(name)]
    if len(folders) > 0:
        for folder in folders:
            file_count = save_yolo_format(folder, outdir, file_count, files_without)
    else:
        file_count = save_yolo_format(source, outdir, file_count, files_without)

    print(f'{files_without} files without annotation')
    print(f'{file_count} files created')



