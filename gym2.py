import os
import shutil
import random

# Define the paths
source_dir = r"C:\Users\kofie\Desktop\semantic-seg-deeplab\data\modified\anno"
train_dir = r"C:\Users\kofie\Desktop\semantic-seg-deeplab\data\modified\train_anno"
test_dir = r"C:\Users\kofie\Desktop\semantic-seg-deeplab\data\modified\test_anno"

# Set a random seed for reproducibility
random.seed(42)

# Create train and test directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of .png and .jpeg files in the source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.bmp', '.png', '.jpeg'))]

# Sort files for consistent ordering
files.sort()

# Shuffle files randomly
random.shuffle(files)

# Split files into 80% train and 20% test
split_index = int(len(files) * 0.8)
train_files = files[:split_index]
test_files = files[split_index:]

# Copy files to the respective directories
for file in train_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

for file in test_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

print(f"Files have been split into {len(train_files)} for training and {len(test_files)} for testing.")
