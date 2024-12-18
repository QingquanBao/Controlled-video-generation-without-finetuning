import os
import cv2
from tqdm import tqdm

# Create the output folder if it doesn't exist
if not os.path.exists('golf_image_data'):
    os.makedirs('golf_image_data')

# Iterate over all videos in the 'videos_160' folder
for filename in tqdm(os.listdir('videos_160')):
    if filename.endswith(".mp4") or filename.endswith(".avi"):
        # Open the video file
        cap = cv2.VideoCapture(os.path.join('videos_160', filename))
        
        # Check if the video has less than 300 frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 300:
            # Subsample the video by 2
            subsampled_frames = []
            for i in range(frame_count):
                ret, frame = cap.read()
                if i % 2 == 0:
                    subsampled_frames.append(frame)
            
            # Save the subsampled frames as images
            for i, frame in enumerate(subsampled_frames):
                if i < frame_count / 2 / 4 or i > frame_count / 2 * 4 / 5:
                    continue
                cv2.imwrite(os.path.join('golf_image_data', f"{filename}_{i}.jpg"), frame)
        
        # Release the video capture
        cap.release()

import shutil
import random

# Get the list of files in the 'golf_image_data' folder
files = os.listdir('golf_image_data')

# Randomly split the files into train and validation sets (4:1 ratio)
train_files = random.sample(files, int(len(files) * 0.8))
val_files = [file for file in files if file not in train_files]

# Create the train and validation folders if they don't exist
if not os.path.exists('golf_image_data/train'):
    os.makedirs('golf_image_data/train')
if not os.path.exists('golf_image_data/validation'):
    os.makedirs('golf_image_data/validation')

# Move the files to their respective folders
for file in train_files:
    shutil.move(os.path.join('golf_image_data', file), 'golf_image_data/train')
for file in val_files:
    shutil.move(os.path.join('golf_image_data', file), 'golf_image_data/validation')


