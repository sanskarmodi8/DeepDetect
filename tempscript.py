import os
import shutil
from pathlib import Path

# Define your root directory and data_path where 'manipulated' and 'original' directories are located
root_dir = "/home/sanskar-modi/current_working_personal_projects/DeepDetect/artifacts/data_preprocessing"  # Replace with your actual path
data_path = "/home/sanskar-modi/current_working_personal_projects/DeepDetect/artifacts/data_ingestion/data"  # Replace with your actual path

# Define splits
splits = ['train', 'test', 'val']

# Function to get video category (manipulated or original) by checking data_path
def get_video_category(video_name, data_path):
    manipulated_dir = os.path.join(data_path, "manipulated")
    original_dir = os.path.join(data_path, "original")
    
    # Check if video exists in manipulated or original directories
    if os.path.exists(os.path.join(manipulated_dir, video_name)):
        return "fake"
    elif os.path.exists(os.path.join(original_dir, video_name)):
        return "original"
    else:
        raise FileNotFoundError(f"Video {video_name} not found in data_path folders.")

# Create new folder structure and move videos
for split in splits:
    split_dir = os.path.join(root_dir, split)
    original_dir = os.path.join(split_dir, "original")
    fake_dir = os.path.join(split_dir, "fake")
    
    # Create directories if they don't exist
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Iterate through videos in the split directory
    for video_file in os.listdir(split_dir):
        video_path = os.path.join(split_dir, video_file)
        
        # Skip directories (we are only looking for files)
        if os.path.isdir(video_path):
            continue
        
        try:
            # Determine whether the video is 'original' or 'fake'
            category = get_video_category(video_file, data_path)
            
            # Move video to the appropriate folder
            if category == "original":
                shutil.move(video_path, os.path.join(original_dir, video_file))
            elif category == "fake":
                shutil.move(video_path, os.path.join(fake_dir, video_file))
            
            print(f"Moved {video_file} to {category} in {split}")
        
        except FileNotFoundError as e:
            print(f"Error: {e}")

print("Reorganization complete!")
