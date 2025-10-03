import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from pyfiglet import Figlet

def save_clip(clip_num, start_frame, end_frame, video_path, out_dir, top_trim=None, bottom_trim=None):
    cap = cv2.VideoCapture(video_path)
    filename = f'{out_dir}/clip{clip_num}.mp4'
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    original_height = height
    if top_trim:
        height -= top_trim
    if bottom_trim:
        height -= original_height - bottom_trim
    writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    valid, f = cap.read()
    count = start_frame
    print(f"Saving {end_frame - start_frame} to {filename}")
    with tqdm(total=end_frame - start_frame) as pbar:
        while valid and count < end_frame:
            if top_trim:
                f = f[top_trim:, :, :]
            if bottom_trim:
                trim_amt = original_height - bottom_trim
                f = f[:-trim_amt, :, :]
            writer.write(f)
            count += 1
            pbar.update(1)
            valid, f = cap.read()
    print(f'Video saved')
    writer.release()
    cap.release()

def draw_text(x, y, text, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 / 2
    color = (0, 255, 0)
    thickness = 1
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_horizontal_line(line_height, img, text):
    img_width = img.shape[1]
    cv2.line(img, (0, line_height), (width, line_height), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 1
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    # Center Horizontally
    text_x = (img_width - text_size[0]) // 2
    # Draw slightly above line
    text_y = line_height - 10
    cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 255, 0), thickness)
    return img


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--video_path", required=True, type=Path)
    arg_parser.add_argument("--output_dir", required=True, type=Path)
    arg_parser.add_argument("--trim_bottom_at", required=False, type=int, default=0)
    arg_parser.add_argument("--trim_top_at", required=False, type=int, default=0)
    args = arg_parser.parse_args()

    fig = Figlet(font='big')
    print(fig.renderText("Video Clipper"))
    print("=" * 12)
    print("= Controls =")
    print("="* 12)
    print("s => Save clip")
    print("Space => Play / Stop")
    print("r => Reset Cursor")
    print("j => Jump to Frame")
    print("ESC => Quit")
    print("Right Arrow => Next Frame")
    print("Left Arrow => Previous Frame")

    input_video = args.video_path
    output_directory = args.output_dir
    trim_bottom_at = args.trim_bottom_at
    trim_top_at = args.trim_top_at

    video = cv2.VideoCapture(str(input_video))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = video.read()
    frame_count = 0
    start_frame = frame_count
    end_frame = None
    play = False

    num_clips = len(list(Path(output_directory).glob("*.mp4")))
    print(f'\n Found {num_clips} existing clips.')
    while ret:
        if trim_top_at > 0:
            draw_horizontal_line(trim_top_at, frame, f'Top')
        if trim_bottom_at > 0:
            draw_horizontal_line(trim_bottom_at, frame, f'Bottom')
        draw_text(10, 30, f'Frame {frame_count} / {num_frames}', frame)
        draw_text(10, 50, f'{num_clips} Clip(s)', frame)
        draw_text(10, 80, f'Start Frame {start_frame}', frame)
        draw_text(10, 100, f'{frame_count - start_frame} frames selected', frame)
        cv2.imshow(input_video.name, frame)
        if not play:
            key = cv2.waitKeyEx(0)
        else:
            # stop playing
            play = cv2.waitKey(1) & 0xFF != 32
            key = -1

        # Handle key press
        # Right arrow ubuntu
        if key == 65363:
            end_frame = frame_count
            frame_count += 1
        # left arrow ubuntu
        elif key == 65361:
            frame_count -= 1
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        # Continue playing
        elif key == -1:
            frame_count += 1
        # Play
        elif key == 32:
            play = True
        # save clip
        elif key == ord('s'):
            if start_frame > frame_count:
                print("Start frame is ahead of frame pointer, resetting frame pointer and try again.")
            save_clip(num_clips, start_frame, frame_count, input_video, output_directory, trim_top_at, trim_bottom_at)
            num_clips += 1
            frame_count += 1
            start_frame = frame_count
        elif key == ord('r'):
            start_frame = frame_count
        # Esc ubuntu
        elif key == ord('j'):
            try:
                jump_frame = int(input("Jumping to frame: "))
                if jump_frame > video.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("Invalid Frame")
                frame_count = jump_frame
                video.set(cv2.CAP_PROP_POS_FRAMES, jump_frame)
            except Exception as e:
                print("Invalid Frame")
                continue
        elif key == 27:
            print(f'Exiting on frame: {frame_count} / {num_frames}')
            break
        else:
            print(f'Key value: {key} not recognized')
            frame_count += 1
        ret, frame = video.read()
    video.release()
    cv2.destroyAllWindows()