import os
import cv2
import random

"""
This code randomly selects a certain 
number of frames from each video in the 
videos folder for training the artificial intelligence 
and saves them in both 720p and 1080p resolutions for the AI to learn.
"""


def clear_folder(folder_path):
    """
    Deletes all files in the specified folder. This is useful for ensuring that
    the output folders start empty before saving new frames.

    Parameters:
    folder_path (str): Path to the folder to be cleared.
    """
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def convert_frames_to_resolution(video_folder, output_folder_720p, output_folder_1080p, total_frames=300):
    """
    Converts a specified number of random frames from videos in a folder to both 720p and 1080p resolutions,
    and saves them. The process involves resizing frames to two different resolutions and storing them
    in separate folders.

    Parameters:
    video_folder (str): Path to the folder containing video files.
    output_folder_720p (str): Folder path where 720p frames will be saved.
    output_folder_1080p (str): Folder path where 1080p frames will be saved.
    total_frames (int): Total number of frames to be saved.
    """

    # Ensure output directories exist, create them if they don't
    os.makedirs(output_folder_720p, exist_ok=True)
    os.makedirs(output_folder_1080p, exist_ok=True)

    # Clear existing contents in the output folders to avoid mixing old and new data
    clear_folder(output_folder_720p)
    clear_folder(output_folder_1080p)

    # Find all video files in the specified folder with the expected file extensions
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("No video files found in the folder.")
        return

    # Calculate how many frames to extract per video, and distribute any extra frames across videos
    frames_per_video = total_frames // len(video_files)
    extra_frames = total_frames % len(video_files)
    processed_frame_count = 0

    # Process each video file separately
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}.")
            continue

        # Randomly select frame indices to capture, ensuring no more than the total frames are selected
        frame_indices_to_capture = random.sample(
            range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
            min(frames_per_video + bool(extra_frames), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        )
        extra_frames -= 1

        for frame_idx in frame_indices_to_capture:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                print(f"Error: Could not read frame {frame_idx} from {video_file}.")
                continue

            # Construct a unique filename for each frame
            frame_name = f"{os.path.splitext(video_file)[0]}_frame_{frame_idx}.jpg"

            # Save the frame in 720p resolution
            cv2.imwrite(os.path.join(output_folder_720p, frame_name), cv2.resize(frame, (1280, 720)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            # Save the same frame in 1080p resolution
            cv2.imwrite(os.path.join(output_folder_1080p, frame_name), cv2.resize(frame, (1920, 1080)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            processed_frame_count += 1
            if processed_frame_count >= total_frames:
                break

        cap.release()

        if processed_frame_count >= total_frames:
            break

    print(f"Processed {processed_frame_count} frames from videos.")


# Example usage
video_path = "video_datas/"  # Replace with your actual video folder path
output_folder_720p = "datas/720p_frames"  # Replace with your desired output folder path for 720p frames
output_folder_1080p = "datas/1080p_frames"  # Replace with your desired output folder path for 1080p frames
total_frame_count=300

convert_frames_to_resolution(video_path, output_folder_720p, output_folder_1080p, total_frame_count)
