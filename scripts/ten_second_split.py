"""
USAGE: USE THIS SCRIPT TO SPLIT VIDEOS INTO 10 SECONDS SEGMENTS

This file is a Python script that segments a video into multiple videos of a specified duration.
It uses the OpenCV library to read and write video files, and the os and tqdm libraries for file and progress bar
operations, respectively.

The segment_video function takes an input video file, an output video file pattern, and an optional segment duration as
input. It opens the input video file using OpenCV's VideoCapture class, and then calculates the frame rate and total
number of frames in the video. It then calculates the number of frames for the segment duration, and initializes the
current frame and segment index. It then loops through the video frames and segments the video into multiple videos of
the specified duration. For each segment, it creates a VideoWriter object to write the segment to a new video file, and
then writes the frames to the output file. Finally, it releases the VideoWriter object and increments the segment index
for the next segment.

The main part of the script demonstrates how to use the segment_video function to segment all input videos in a
specified directory. It first specifies the input and output directories, and then loops over all input videos in the
input directory. For each video, it calls the segment_video function to segment the video into multiple videos of the
specified duration, and saves the segmented videos to the output directory.

To use this script, you can simply replace the input_dir and output_dir variables with the paths to your input and
output directories, and then run the script. It will segment all videos in the input directory and save the segmented
videos to the output directory.

WARNING: Some videos may have less than 10 seconds of content, so the last segment may be shorter than the specified
duration. You may need to handle this case separately if necessary.
"""

import cv2
import os
from tqdm import tqdm


def segment_video(in_file, out_file, segment_duration=10):
    """
    This function segments a video into multiple videos of a specified duration.
    :param in_file: The input video file.
    :param out_file: The output video file pattern (e.g., 'output_%03d.mp4').
    :param segment_duration: The duration of each segment in seconds (default is 10 seconds).
    :return:
    """
    # Open the input video file
    cap = cv2.VideoCapture(in_file)
    # Get the frame rate and total number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames for the segment duration
    # e.g., 10 seconds at 30 fps is 300 frames (10 * 30 = 300)
    segment_frames = int(segment_duration * fps)

    # Initialize the current frame and segment index
    current_frame = 0
    segment_index = 0

    # Loop through the video frames and segment the video
    while current_frame < total_frames:
        # Set the start and end frames for the segment (e.g., 0 - 300, 300 - 600, etc.)
        start_frame = current_frame
        # Ensure the end frame does not exceed the total number of frames
        # e.g., for the last segment, the end frame should be the total number of frames
        end_frame = min(start_frame + segment_frames, total_frames)

        # Set the output file name for the segment (e.g., output_000.mp4, output_001.mp4, etc.)
        output_file = out_file % segment_index

        # Create a VideoWriter object to write the segment to a new video file
        # The VideoWriter object takes the output file name, codec, frame rate, and frame size as input
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

        # Set the current frame to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Loop through the frames and write them to the output file
        while current_frame < end_frame:
            # Read the next frame from the input video
            ret, frame = cap.read()
            if not ret:
                break
            # Write the frame to the output video, writes in the location of the output file
            out.write(frame)
            current_frame += 1

        out.release()
        # Increment the segment index for the next segment
        segment_index += 1

    cap.release()


if __name__ == '__main__':
    # Specify the input directory (containing the videos to be segmented)
    input_dir = '/path/to/folder'
    # Specify the output directory (where the segmented videos will be saved)
    output_dir = '/path/to/output/folder'
    # segment_duration = 10  # seconds

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all input videos in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".mp4"):
            input_file = os.path.join(input_dir, filename)
            output_pattern = os.path.join(output_dir, filename.split('.')[0] + '_%03d.mp4')
            segment_video(input_file, output_pattern)
