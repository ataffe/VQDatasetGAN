import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from pyfiglet import Figlet
import copy
from multiprocessing import Process

def save_clip(clip_num, start_frame, end_frame, video_path, out_dir, top_trim=None, bottom_trim=None):
    cap = cv2.VideoCapture(video_path)
    vid_name = Path(video_path).stem
    filename = f'{out_dir}/surgvu{vid_name}_clip{clip_num}.mp4'
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
    cv2.line(img, (195, line_height), (width-195, line_height), (0, 255, 0), 2)
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

def draw_edit_box(f):
    cv2.rectangle(f, (5, 120), (250, 210), (0, 255, 0), 1)
    draw_text(10, 140, f'Editing Mode Enabled!', f)
    draw_text(10, 160, f'Press t to select top line.', f)
    draw_text(10, 180, f'Press b to select bottom line.', f)
    draw_text(10, 200, f'Press e to toggle edit mode.', f)

def draw_editing_line_message(f, top=True):
    cv2.rectangle(f, (5, 120), (250, 230), (0, 255, 0), 1)
    draw_text(10, 140, f'Editing Mode Enabled!', f)
    if top:
        draw_text(10, 160, f'Editing top line!', f)
    else:
        draw_text(10, 160, f'Editing bottom line!', f)
    draw_text(10, 180, f'Press w to move line up.', f)
    draw_text(10, 200, f'Press z to move line down.', f)
    draw_text(10, 220, f'Press e to toggle edit mode.', f)

def draw_seeking_box(f):
    cv2.rectangle(f, (5, 120), (200, 150), (0, 255, 0), 1)
    draw_text(10, 140, f'Seeking Mode Enabled!', f)


def get_total_output_frames(outdir):
    video_clips = list(Path(outdir).rglob('*.mp4'))
    total_frames = 0
    for video_clip in video_clips:
        vid = cv2.VideoCapture(str(video_clip))
        total_frames += vid.get(cv2.CAP_PROP_FRAME_COUNT)
        vid.release()
    return total_frames

def cleanup_processes(processes):
    for idx, process in enumerate(processes):
        if not process.is_alive():
            processes.remove(process)
    return processes

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--video_dir", required=True, type=Path)
    arg_parser.add_argument("--output_dir", required=True, type=Path)
    arg_parser.add_argument("--trim_bottom_at", required=False, type=int, default=0)
    arg_parser.add_argument("--trim_top_at", required=False, type=int, default=0)
    arg_parser.add_argument("--start_at_video_num", required=False, type=int, default=0)
    args = arg_parser.parse_args()

    fig = Figlet(font='big')
    print(fig.renderText("Video Clipper"))
    print("=" * 12)
    print("= Controls =")
    print("="* 12)
    print("s => Save clip")
    print("Space => Toggle Play / Stop")
    print("d => move forward")
    print("a => move backward")
    print("r => Reset Cursor")
    print("n => Next Video")
    print("j => Jump to Frame")
    print("e => Toggle Edit Mode")
    print("f => Toggle Seek Mode")
    print("ESC => Quit")

    video_dir = args.video_dir
    output_directory = args.output_dir
    trim_bottom_at = args.trim_bottom_at
    trim_top_at = args.trim_top_at
    start_at_video_num = args.start_at_video_num
    saving_processes = []

    videos = list(Path(video_dir).rglob("*.mp4"))
    print(f"Found {len(videos)} videos")
    exit_all = False
    total_dataset_frames = get_total_output_frames(output_directory)
    print(f"Total frames: {total_dataset_frames}")
    process_cleanup_ctr = 0
    num_videos = len(videos)
    for idx, video_path in enumerate(videos):
        if idx < start_at_video_num:
            continue
        video_path = Path(video_path)
        if exit_all:
            break
        video = cv2.VideoCapture(str(video_path))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = video.read()
        frame_count = 0
        start_frame = frame_count
        end_frame = None
        play = False
        seeking = False
        editing_line = False
        editing_top_line = False
        editing_bottom_line = False
        right_arrow_val = 65363
        left_arrow_val = 65361
        up_arrow_val = 65362
        down_arrow_val = 65364
        unedited_frame = copy.deepcopy(frame)
        num_clips = len([f for f in Path(output_directory).rglob("*.mp4") if video_path.stem in str(f)])
        cv2.namedWindow(video_path.name)
        def nothing(x):
            pass

        cv2.createTrackbar('seek_bar', video_path.name, 0, num_frames, nothing)
        while ret:
            # Drawing elements on left and right.
            if trim_top_at > 0:
                 draw_horizontal_line(trim_top_at, frame, f'Top')
            if trim_bottom_at > 0:
                draw_horizontal_line(trim_bottom_at, frame, f'Bottom')
            draw_text(10, 30, f'Frame {frame_count} / {num_frames}', frame)
            # draw_text(10, 50, f'{num_clips} Clip(s)', frame)
            draw_text(10, 60, f'Start Frame {start_frame}', frame)
            draw_text(10, 80, f'{frame_count - start_frame} frames selected', frame)

            draw_text(width - 190, 30, f'Total frames:{total_dataset_frames:,}', frame)
            draw_text(width - 190, 70, f'{num_clips} Clip(s)', frame)
            draw_text(width - 190, 90, f'video: {idx} / {num_videos}', frame)

            if editing_line and not (editing_top_line or editing_bottom_line):
                draw_edit_box(frame)
            elif editing_top_line:
                draw_editing_line_message(frame)
            elif editing_bottom_line:
                draw_editing_line_message(frame, top=False)
            elif seeking:
                draw_seeking_box(frame)

            cv2.imshow(video_path.name, frame)

            # Wait for key press
            if not play and not seeking:
                key = cv2.waitKeyEx(0)
            elif play and not seeking:
                # stop playing
                play = cv2.waitKey(1) & 0xFF != 32 # Space bar toggle
                if not play:
                    cv2.setTrackbarPos('seek_bar', video_path.name, frame_count)
                key = -1
            elif seeking:
                if cv2.waitKey(1) & 0xFF == ord('f'):
                    key = ord('f')
                elif frame_count != cv2.getTrackbarPos('seek_bar', video_path.name):
                    new_pos = cv2.getTrackbarPos('seek_bar', video_path.name)
                    video.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    ret, frame = video.read()
                    frame_count = new_pos
                    unedited_frame = copy.deepcopy(frame)
                    continue
                else:
                    frame = copy.deepcopy(unedited_frame)
                    continue
            else:
                key = -99

            ################################# Handle key press ##############################
            # Next
            if key == ord('d'):
                if frame_count < num_frames:
                    end_frame = frame_count
                    frame_count += 5
                    cv2.setTrackbarPos('seek_bar', video_path.name, frame_count)
                else:
                    frame = copy.deepcopy(unedited_frame)
                    continue
            # Previous
            elif key == ord('a'):
                if frame_count > 0:
                    frame_count -= 5
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    cv2.setTrackbarPos('seek_bar', video_path.name, frame_count)
                else:
                    frame = copy.deepcopy(unedited_frame)
                    continue
            # Continue playing
            elif key == -1:
                frame_count += 1
            # Play
            elif key == 32:
                play = True
            # Toggle Seeking
            elif key == ord('f'):
                seeking = not seeking
            # save clip
            elif key == ord('s'):
                if start_frame > frame_count:
                    print("Start frame is ahead of frame pointer, resetting frame pointer and try again.")
                # Kick off a new process to save the clip
                saving_processes.append(Process(
                    target=save_clip,
                    args=(num_clips,
                          start_frame,
                          frame_count,
                          str(video_path),
                          output_directory,
                          trim_top_at,
                          trim_bottom_at)))
                saving_processes[-1].start()
                num_clips += 1
                frame_count += 1
                start_frame = frame_count
                total_dataset_frames = get_total_output_frames(output_directory)
                if process_cleanup_ctr == 10:
                    saving_processes = cleanup_processes(saving_processes)
                    process_cleanup_ctr = 0
            # Toggle edit mode
            elif key == ord('e'):
                if editing_line:
                    editing_top_line = False
                    editing_bottom_line = False
                    frame = copy.deepcopy(unedited_frame)
                editing_line = not editing_line
                continue
            # Edit Top line
            elif key == ord('t'):
                if editing_line:
                    editing_top_line = True
                    frame = copy.deepcopy(unedited_frame)
                continue
            # Edit bottom line
            elif key == ord('b'):
                if editing_line:
                    editing_bottom_line = True
                    frame = copy.deepcopy(unedited_frame)
                continue
            # move line up
            elif key == ord('w'):
                if editing_line:
                    if editing_top_line:
                        trim_top_at -= 1
                    elif editing_bottom_line:
                        trim_bottom_at -= 1
                frame = copy.deepcopy(unedited_frame)
                continue
            # move line down
            elif key == ord('z'):
                if editing_line:
                    if editing_top_line:
                        trim_top_at += 1
                    elif editing_bottom_line:
                        trim_bottom_at += 1
                frame = copy.deepcopy(unedited_frame)
                continue
            # r => Reset Start Frame
            elif key == ord('r'):
                start_frame = frame_count
            # n = Next video
            elif key == ord('n'):
                break
            # j => Jump to frame
            elif key == ord('j'):
                try:
                    jump_frame = cv2.getTrackbarPos('seek_bar', video_path.name)
                    frame_count = jump_frame
                    video.set(cv2.CAP_PROP_POS_FRAMES, jump_frame)
                except Exception as e:
                    print("Invalid Frame")
                    continue
            # Esc => Exit
            elif key == 27:
                print(f'Exiting on frame: {frame_count} / {num_frames}')
                exit_all = True
                break
            else:
                frame = copy.deepcopy(unedited_frame)
                continue
                # print(f'Key value: {key} not recognized')
            ################################# Handle key press ##############################
            ret, frame = video.read()
            if not play:
                unedited_frame = copy.deepcopy(frame)
        video.release()
        cv2.destroyAllWindows()
    if len(saving_processes) > 0:
        print("Finishing Saving Clips...")
        try:
            for process in saving_processes:
                process.join()
        except KeyboardInterrupt:
            print("Terminating saving processes")
            for process in saving_processes:
                process.terminate()
                process.join()