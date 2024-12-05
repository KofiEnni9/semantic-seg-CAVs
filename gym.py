import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the original mask image
mask = Image.open(r"C:\Users\kofie\Desktop\real_world_data\Dataset4_NorthSlope_Segmentation\Dataset1\annotations\north_slope_video.mp4_0_1.jpg").convert('RGB')
mask_np = np.array(mask)

# Reshape mask to a 2D array of pixels (each row is an RGB color)
pixels = mask_np.reshape(-1, 3)

# Perform K-means clustering with 6 clusters
n_classes = 5
kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(pixels)

# Get the cluster centers (quantized colors)
quantized_colors = kmeans.cluster_centers_.astype(int)
print("Quantized Colors (Cluster Centers):", quantized_colors)

# Replace each pixel with its cluster's center color
quantized_pixels = quantized_colors[kmeans.labels_]
quantized_image_np = quantized_pixels.reshape(mask_np.shape)

# Convert back to an image for visualization
quantized_image = Image.fromarray(quantized_image_np.astype('uint8'))

# Display or save the quantized image
quantized_image.show()  # Opens the image in the default viewer

# Alternatively, to save the image, uncomment the line below
# quantized_image.save('quantized_mask_preview.png')
