import os
import shutil

def rename_and_move_images(source_folder, target_folder):
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Get list of all files in the source folder
    files = os.listdir(source_folder)
    
    # Filter for .png and .jpg files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg'))]
    
    for i, file_name in enumerate(image_files, start=780):
        # Construct full file paths
        source_path = os.path.join(source_folder, file_name)
        file_extension = os.path.splitext(file_name)[1]  # Get file extension (e.g., '.png')
        new_name = f"img_{i}{file_extension}"  # Create new file name
        target_path = os.path.join(target_folder, new_name)
        
        # Move and rename the file
        shutil.copy(source_path, target_path)
        print(f"Moved: {source_path} -> {target_path}")

# Example usage:
source_folder = r"C:\Users\kofie\Desktop\real_world_data\Dataset3_NorthFarm_Segmentation\raw_images"  # Replace with your source folder path
target_folder = r"C:\Users\kofie\Desktop\tempdata\img2"  # Replace with your target folder path
rename_and_move_images(source_folder, target_folder)
