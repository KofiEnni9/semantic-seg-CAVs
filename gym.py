import os
import cv2
import numpy as np
import torch

# Preprocessing code regerate annos since there were not always good

# Define paths
path_of_images = r"C:\Users\kofie\Desktop\real_world_data\Dataset3_NorthFarm_Segmentation\annotations"
infer_to_path = r"C:\Users\kofie\Desktop\real_world_data\Dataset3_NorthFarm_Segmentation\annotations_remix"

# Define target and class colors
target_colors = {
    0: (0, 0, 0),
    1: [(208, 254, 157), (172, 208, 69)],  # Light green variants
    2: [(59, 93, 4), (3, 48, 0)],         # Dark green variants
    3: (155, 155, 154),                   # Gray
    4: (138, 87, 42),                     # Brown
    5: (183, 21, 123),                    # Pink
    6: (73, 143, 225),                   # Blue
    7: (195, 234, 254)           #water
}

class_colors = {
    0: (0, 0, 0),       # Black
    1: (208, 254, 157), # Light green
    2: (59, 93, 4),     # Dark green
    3: (155, 155, 154), # Gray
    4: (138, 87, 42),   # Brown
    5: (183, 21, 123),  # Pink
    6: (73, 143, 225),   # Blue 
    7: (195, 234, 254)    #water
}

import numpy as np
from collections import deque

def mask_fix(mask):
    """
    Replace zeros in the mask with the nearest non-zero value using an efficient multi-source BFS approach.
    """
    if mask is None:
        return None
    
    height, width = mask.shape
    new_mask = mask.copy()
    queue = deque()
    
    # Initialize the queue with all non-zero positions and store their values
    for y in range(height):
        for x in range(width):
            if mask[y, x] != 0:
                queue.append((y, x))
    
    # Define directions for BFS: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS to propagate nearest non-zero values
    while queue:
        y, x = queue.popleft()
        current_value = new_mask[y, x]
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and new_mask[ny, nx] == 0:
                new_mask[ny, nx] = current_value
                queue.append((ny, nx))
    
    return new_mask


# Function to load annotation
def load_annotation(path, target_colors):
    """Load and process annotation mask"""
    annotation = cv2.imread(path, cv2.IMREAD_COLOR)
    if annotation is None:
        print(f"Failed to load annotation: {path}")
        return None

    annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2RGB)
    mask = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.int64)

    annotation_reshaped = annotation.reshape(-1, 3)
    mask_reshaped = np.zeros(annotation_reshaped.shape[0], dtype=np.int64)

    tolerance = 5 # Tolerance for color matching

    for class_idx, color in target_colors.items():
        if isinstance(color, tuple):  # Single color
            within_tolerance = np.all(np.abs(annotation_reshaped - color) <= tolerance, axis=1)
        elif isinstance(color, list):  # Color range
            color1, color2 = color
            within_tolerance1 = np.all(np.abs(annotation_reshaped - color1) <= tolerance, axis=1)
            within_tolerance2 = np.all(np.abs(annotation_reshaped - color2) <= tolerance, axis=1)
            within_tolerance = np.logical_or(within_tolerance1, within_tolerance2)

        mask_reshaped[within_tolerance] = class_idx

    mask = mask_reshaped.reshape(annotation.shape[:2])
    return mask

# Function to colorize and save images
def inferring_img(mask, img_name, infer_path):
    def colorize_mask(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().astype(np.uint8)

        height, width = mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx, color in class_colors.items():
            colored_mask[mask == class_idx] = color

        return colored_mask

    colored_original = colorize_mask(mask)
    the_bgr = cv2.cvtColor(colored_original, cv2.COLOR_RGB2BGR)
    original_save_path = os.path.join(infer_path, f'{img_name}')
    cv2.imwrite(original_save_path, the_bgr)

# Process each image in the directory
if not os.path.exists(infer_to_path):
    os.makedirs(infer_to_path)

image_files = [f for f in os.listdir(path_of_images) if f.endswith(('.png', '.jpg', '.jpeg'))]
for image_file in image_files:
    image_path = os.path.join(path_of_images, image_file)
    mask = load_annotation(image_path, target_colors)

    mask = mask_fix(mask)

    if mask is not None:
        inferring_img(mask, image_file, infer_to_path)
        print(f"Processed and saved: {image_file}")
    else:
        print(f"Skipping image due to load failure: {image_file}")
