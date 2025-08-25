import argparse

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=Path, required=True)
    parser.add_argument('--output_folder', type=Path, required=True)
    parser.add_argument('--images_per_video', type=int, default=50000)
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    output_folder.mkdir(exist_ok=True, parents=True)
    video_files = list(input_folder.glob('*.mp4'))
    print(f'Found {len(video_files)} videos')
    debt = 0
    images_per_video = args.images_per_video
    for video_file in video_files:
        video = cv2.VideoCapture(str(video_file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = video_file.name
        ret, frame = video.read()
        frame_count = 0
        debt += images_per_video - num_frames if num_frames < images_per_video else 0
        frames_to_process = images_per_video
        if num_frames < frames_to_process:
            frames_to_process = num_frames
            debt += images_per_video - num_frames
        elif debt > 0 and debt + frames_to_process <= num_frames:
            frames_to_process += debt
            debt = 0
        elif debt > 0 and debt + frames_to_process > num_frames:
            frames_to_process = num_frames
            debt -= frames_to_process - images_per_video

        print(f'Processing {video_name} with frame count {num_frames}')
        for _ in tqdm(range(frames_to_process), total=frames_to_process, desc=f'Processing {video_name}', unit='frames'):
            if not ret:
                raise RuntimeError("Error invalid frame: {}".format(frame_count))
            if np.sum(frame) < 100:
                print(f'Skipping frame {output_folder}/{video_name}_{frame_count}.jpg')
                ret, frame = video.read()
                continue
            filename = f'{output_folder}/{video_name}_{frame_count}.jpg'
            frame = cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, frame)
            frame_count += 1
            ret, frame = video.read()



