import argparse

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=Path, required=True)
    parser.add_argument('--output_folder', type=Path, required=True)
    parser.add_argument("--prefix", type=str, default='', required=False)
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    output_folder.mkdir(exist_ok=True, parents=True)
    video_files = list(input_folder.glob('*.mp4'))
    prefix = args.prefix
    print(f'Found {len(video_files)} videos')
    for video_file in video_files:
        video = cv2.VideoCapture(str(video_file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = video_file.name
        ret, frame = video.read()
        frame_count = 0
        print(f'Processing {video_name} with frame count {num_frames}')
        for _ in tqdm(range(num_frames), total=num_frames, desc=f'Processing {video_name}', unit='frames'):
            if not ret:
                raise RuntimeError("Error invalid frame: {}".format(frame_count))
            if np.sum(frame) < 100:
                print(f'Skipping frame {output_folder}/{prefix}-{video_name}_{frame_count}.jpg')
                ret, frame = video.read()
                continue
            filename = f'{output_folder}/surgvu{video_name}_{frame_count}.jpg'
            frame = cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, frame)
            frame_count += 1
            ret, frame = video.read()



