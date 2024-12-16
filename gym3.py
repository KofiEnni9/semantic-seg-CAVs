# import numpy as np
# from PIL import Image
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Load the original mask image
# mask = Image.open(r"C:\Users\kofie\Desktop\real_world_data\Dataset3_NorthFarm_Segmentation\annotations\1570151446.872610832.png").convert('RGB')
# mask_np = np.array(mask)

# # Reshape mask to a 2D array of pixels (each row is an RGB color)
# pixels = mask_np.reshape(-1, 3)

# # Perform K-means clustering with 6 clusters
# n_classes = 8
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(pixels)

# # Get the cluster centers (quantized colors)
# quantized_colors = kmeans.cluster_centers_.astype(int)
# print("Quantized Colors (Cluster Centers):", quantized_colors)

# # Replace each pixel with its cluster's center color
# quantized_pixels = quantized_colors[kmeans.labels_]
# quantized_image_np = quantized_pixels.reshape(mask_np.shape)

# # Convert back to an image for visualization
# quantized_image = Image.fromarray(quantized_image_np.astype('uint8'))

# # Display or save the quantized image
# quantized_image.show()  # Opens the image in the default viewer

# Alternatively, to save the image, uncomment the line below
# quantized_image.save('quantized_mask_preview.png')


import os
import cv2
import numpy as np
import torch

# Preprocessing code regerate annos since there were not always good

# Define paths
infer_to_path = 'data/ishere'

# Define target and class colors
target_colors = {
    0: (0, 0, 0),
    1: [(208, 254, 157), (172, 208, 69)],  # Light green variants
    2: [(59, 93, 4), (3, 48, 0)],         # Dark green variants
    3: (155, 155, 154),                   # Gray
    4: (138, 87, 42),                     # Brown
    5: (183, 21, 123),                    # Pink
    6: (73, 143, 225)                     # Blue
}

class_colors = {
    0: (0, 0, 0),       # Black
    1: (208, 254, 157), # Light green
    2: (59, 93, 4),     # Dark green
    3: (155, 155, 154), # Gray
    4: (138, 87, 42),   # Brown
    5: (183, 21, 123),  # Pink
    6: (73, 143, 225)   # Blue 
}

import numpy as np
from collections import deque

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
def inferring_img(mask, infer_path):
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
    original_save_path = os.path.join(infer_path, f'img_okkay.png')
    cv2.imwrite(original_save_path, the_bgr)

# Process each image in the directory
if not os.path.exists(infer_to_path):
    os.makedirs(infer_to_path)

image_path = r"C:\Users\kofie\Desktop\real_world_data\Train\annos\anno_471.png"
mask = load_annotation(image_path, target_colors)


if mask is not None:
    inferring_img(mask, infer_to_path)
    print(f"Processed and saved: ")
else:
    print(f"Skipping image due to load failure: ")
